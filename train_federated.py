from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from compression import compress_gradient, estimate_communication, flatten_tensors, top_k_sparsify, unflatten_tensor
from data import make_hospital_loaders
from model import get_model


def get_parameters(model: nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().cpu().clone() for parameter in model.parameters()]


def set_parameters(model: nn.Module, parameters: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for target, source in zip(model.parameters(), parameters):
            target.copy_(source.to(target.device))


def train_one_epoch(model: nn.Module, loader, device: torch.device, lr: float) -> float:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        total += labels.size(0)
    return total_loss / max(1, total)


@torch.no_grad()
def evaluate_auc(model: nn.Module, loaders, device: torch.device) -> float:
    model.eval()
    y_true: list[int] = []
    y_score: list[float] = []
    for loader in loaders:
        for images, labels in loader:
            probs = torch.softmax(model(images.to(device)), dim=1)[:, 1].cpu()
            y_score.extend(probs.tolist())
            y_true.extend(labels.tolist())
    if len(set(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def aggregate_updates(
    global_parameters: list[torch.Tensor],
    client_parameters: list[list[torch.Tensor]],
    client_sizes: list[int],
    mode: str,
    k_ratio: float,
    server_lr: float,
    sign_server_lr: float,
    residuals: list[torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    flat_global, shapes = flatten_tensors(global_parameters)
    weighted_update = torch.zeros_like(flat_global)
    total_size = float(sum(client_sizes))

    for client_id, (parameters, size) in enumerate(zip(client_parameters, client_sizes)):
        flat_client, _ = flatten_tensors(parameters)
        update = flat_client - flat_global
        corrected_update = update
        if residuals is not None and mode.endswith("_ef"):
            corrected_update = update + residuals[client_id]

        if mode == "topk":
            compressed_update = compress_gradient(corrected_update, k_ratio=k_ratio, use_topk=True, use_sign=False)
        elif mode == "topk_ef":
            compressed_update = compress_gradient(corrected_update, k_ratio=k_ratio, use_topk=True, use_sign=False)
            residuals[client_id] = corrected_update - compressed_update
        elif mode == "topk_sign":
            compressed_update = compress_gradient(corrected_update, k_ratio=k_ratio, use_topk=True, use_sign=True)
        elif mode == "topk_sign_ef":
            sparse = top_k_sparsify(corrected_update, k_ratio=k_ratio)
            scale = sparse[sparse != 0].abs().mean().clamp_min(1e-8)
            compressed_update = torch.sign(sparse) * scale
            residuals[client_id] = corrected_update - compressed_update
        else:
            compressed_update = corrected_update

        weighted_update += compressed_update * (size / total_size)

    step_size = sign_server_lr if mode == "topk_sign" else server_lr
    next_flat = flat_global + (step_size * weighted_update)
    return unflatten_tensor(next_flat, shapes)


def run_experiment(args, mode: str) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(args.seed)
    loaders = make_hospital_loaders(
        samples_per_hospital=args.samples_per_hospital,
        batch_size=args.batch_size,
        use_pcam=args.use_pcam,
        seed=args.seed,
    )
    global_model = get_model(args.model).to(device)
    global_parameters = get_parameters(global_model)
    total_params = sum(parameter.numel() for parameter in global_parameters)

    if mode == "fedavg":
        comm = estimate_communication(total_params, k_ratio=1.0, use_sign=False)
    elif mode in {"topk", "topk_ef"}:
        comm = estimate_communication(total_params, k_ratio=args.k_ratio, use_sign=False)
    else:
        comm = estimate_communication(total_params, k_ratio=args.k_ratio, use_sign=True)

    rows: list[dict] = []
    residuals = None
    if mode.endswith("_ef"):
        flat_global, _ = flatten_tensors(global_parameters)
        residuals = [torch.zeros_like(flat_global) for _ in loaders]

    for round_id in range(1, args.rounds + 1):
        client_parameters: list[list[torch.Tensor]] = []
        client_sizes: list[int] = []
        losses: list[float] = []

        for hospital in loaders:
            local_model = deepcopy(global_model).to(device)
            set_parameters(local_model, global_parameters)
            loss = train_one_epoch(local_model, hospital.train, device, lr=args.lr)
            client_parameters.append(get_parameters(local_model))
            client_sizes.append(len(hospital.train.dataset))
            losses.append(loss)

        global_parameters = aggregate_updates(
            global_parameters,
            client_parameters,
            client_sizes,
            mode=mode,
            k_ratio=args.k_ratio,
            server_lr=args.server_lr,
            sign_server_lr=args.sign_server_lr,
            residuals=residuals,
        )
        set_parameters(global_model, global_parameters)
        auc = evaluate_auc(global_model, [hospital.test for hospital in loaders], device)
        row = {
            "mode": mode,
            "round": round_id,
            "auc": auc,
            "loss": float(np.mean(losses)),
            "dense_bytes": comm.dense_bytes,
            "compressed_bytes": comm.compressed_bytes,
            "communication_ratio": comm.ratio,
            "communication_saving": comm.saving,
        }
        rows.append(row)
        print(
            f"{mode:12s} round={round_id:02d} auc={auc:.4f} "
            f"comm={comm.ratio * 100:.3f}% saving={comm.saving * 100:.2f}%"
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Pathology federated learning compression demo")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples-per-hospital", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--server-lr", type=float, default=1.0)
    parser.add_argument("--sign-server-lr", type=float, default=1e-3)
    parser.add_argument("--k-ratio", type=float, default=0.1)
    parser.add_argument("--model", choices=["tiny", "resnet18"], default="tiny")
    parser.add_argument(
        "--mode",
        choices=["fedavg", "topk", "topk_ef", "topk_sign", "topk_sign_ef", "all", "research"],
        default="research",
    )
    parser.add_argument("--use-pcam", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("results.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.mode == "all":
        modes = ["fedavg", "topk", "topk_sign"]
    elif args.mode == "research":
        modes = ["fedavg", "topk", "topk_ef", "topk_sign", "topk_sign_ef"]
    else:
        modes = [args.mode]
    rows: list[dict] = []
    for mode in modes:
        rows.extend(run_experiment(args, mode))
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    summary = (
        df.sort_values("round")
        .groupby("mode", as_index=False)
        .tail(1)
        .sort_values("communication_ratio", ascending=False)
        .loc[:, ["mode", "round", "auc", "communication_ratio", "communication_saving", "compressed_bytes"]]
    )
    summary_path = args.out.with_name("summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"saved {args.out}")
    print(f"saved {summary_path}")


if __name__ == "__main__":
    main()
