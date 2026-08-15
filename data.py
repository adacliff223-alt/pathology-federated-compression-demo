from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms


@dataclass(frozen=True)
class HospitalLoaders:
    name: str
    train: DataLoader
    test: DataLoader
    stain_label: str


class SyntheticPCam(Dataset):
    """Small deterministic pathology-like patch dataset for smoke tests and demos."""

    def __init__(self, samples: int = 1800, image_size: int = 64, seed: int = 7):
        generator = torch.Generator().manual_seed(seed)
        labels = torch.arange(samples) % 2
        labels = labels[torch.randperm(samples, generator=generator)]

        base = torch.rand(samples, 3, image_size, image_size, generator=generator) * 0.18
        base[:, 0] += 0.70
        base[:, 1] += 0.48
        base[:, 2] += 0.66

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, image_size),
            torch.linspace(-1, 1, image_size),
            indexing="ij",
        )
        tumor_mask = ((xx**2 + yy**2) < 0.22).float()
        tumor_mask = tumor_mask.unsqueeze(0).unsqueeze(0)
        texture = torch.rand(samples, 1, image_size, image_size, generator=generator)

        positive = labels.view(-1, 1, 1, 1).float()
        base[:, 0:1] -= positive * tumor_mask * 0.18
        base[:, 1:2] -= positive * tumor_mask * 0.24
        base[:, 2:3] += positive * tumor_mask * 0.22
        base += positive * tumor_mask * (texture - 0.5) * 0.28

        self.images = base.clamp(0.0, 1.0)
        self.labels = labels.long()

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], int(self.labels[index])


class TensorTransformDataset(Dataset):
    def __init__(self, dataset: Dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return self.transform(image), label


def _tensor_stain_transform(brightness: float, contrast: float):
    return transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),
            transforms.ColorJitter(brightness=brightness, contrast=contrast),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
        ]
    )


def _load_base_dataset(use_pcam: bool, samples: int):
    if not use_pcam:
        return SyntheticPCam(samples=samples)

    from torchvision.datasets import PCAM

    return PCAM(root="./data", split="train", download=True, transform=transforms.ToTensor())


def make_hospital_loaders(
    samples_per_hospital: int = 600,
    batch_size: int = 32,
    use_pcam: bool = False,
    seed: int = 7,
) -> list[HospitalLoaders]:
    total = samples_per_hospital * 3
    base = _load_base_dataset(use_pcam=use_pcam, samples=total)
    transforms_by_hospital = [
        ("Hospital A", _tensor_stain_transform(0.20, 0.30), "bright / high contrast"),
        ("Hospital B", _tensor_stain_transform(0.05, 0.10), "mild stain"),
        ("Hospital C", _tensor_stain_transform(0.00, 0.50), "strong contrast"),
    ]

    loaders: list[HospitalLoaders] = []
    generator = torch.Generator().manual_seed(seed)
    for idx, (name, transform, stain_label) in enumerate(transforms_by_hospital):
        start = idx * samples_per_hospital
        subset = Subset(base, range(start, start + samples_per_hospital))
        transformed = TensorTransformDataset(subset, transform)
        train_len = int(samples_per_hospital * 0.8)
        test_len = samples_per_hospital - train_len
        train_ds, test_ds = random_split(transformed, [train_len, test_len], generator=generator)
        loaders.append(
            HospitalLoaders(
                name=name,
                train=DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                test=DataLoader(test_ds, batch_size=batch_size),
                stain_label=stain_label,
            )
        )
    return loaders
