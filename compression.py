from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CompressionStats:
    total_params: int
    transmitted_values: int
    dense_bytes: int
    compressed_bytes: float

    @property
    def ratio(self) -> float:
        return self.compressed_bytes / self.dense_bytes if self.dense_bytes else 0.0

    @property
    def saving(self) -> float:
        return 1.0 - self.ratio


def flatten_tensors(tensors: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Size]]:
    shapes = [tensor.shape for tensor in tensors]
    flat = torch.cat([tensor.detach().reshape(-1) for tensor in tensors])
    return flat, shapes


def unflatten_tensor(vector: torch.Tensor, shapes: list[torch.Size]) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    offset = 0
    for shape in shapes:
        count = int(torch.tensor(shape).prod().item())
        tensors.append(vector[offset : offset + count].reshape(shape))
        offset += count
    return tensors


def top_k_sparsify(vector: torch.Tensor, k_ratio: float = 0.1) -> torch.Tensor:
    if not 0 < k_ratio <= 1:
        raise ValueError("k_ratio must be in (0, 1].")
    k = max(1, int(vector.numel() * k_ratio))
    indices = torch.topk(vector.abs(), k).indices
    sparse = torch.zeros_like(vector)
    sparse[indices] = vector[indices]
    return sparse


def sign_quantize(vector: torch.Tensor) -> torch.Tensor:
    return torch.sign(vector)


def compress_gradient(
    vector: torch.Tensor,
    k_ratio: float = 0.1,
    use_topk: bool = True,
    use_sign: bool = True,
) -> torch.Tensor:
    compressed = top_k_sparsify(vector, k_ratio) if use_topk else vector.clone()
    return sign_quantize(compressed) if use_sign else compressed


def estimate_communication(
    total_params: int,
    k_ratio: float = 1.0,
    use_sign: bool = False,
    value_bits: int = 32,
    index_bits: int = 32,
    include_indices: bool = False,
) -> CompressionStats:
    transmitted = max(1, int(total_params * k_ratio))
    dense_bytes = total_params * 4
    if use_sign:
        value_bytes = transmitted / 8
    else:
        value_bytes = transmitted * (value_bits / 8)
    index_bytes = transmitted * (index_bits / 8) if include_indices and k_ratio < 1.0 else 0
    compressed_bytes = value_bytes + index_bytes
    return CompressionStats(total_params, transmitted, dense_bytes, compressed_bytes)
