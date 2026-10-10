# config.py

from dataclasses import dataclass
from typing import List


@dataclass
class ModelConfig:
    codebook_sizes: List[int]
    max_iter: int = 10
    nredo: int = 1
    max_points_per_centroid: int = 256
    spherical: bool = False
    random_state: int = 42

    def __post_init__(self):
        if len(self.codebook_sizes) == 0:
            raise ValueError("codebook_sizes 不能为空")
        for k in self.codebook_sizes:
            if k <= 1:
                raise ValueError("codebook size 必须 > 1")

    @property
    def num_levels(self) -> int:
        return len(self.codebook_sizes)

    def total_codebook_size(self) -> int:
        total = 1
        for k in self.codebook_sizes:
            total *= k
        return total


@dataclass
class TrainConfig:
    output_dir: str = "./rqkmeans_model"


@dataclass
class DataConfig:
    embedding_dim: int = 1024
    train_samples: int = 20480
    test_samples: int = 200


@dataclass
class Config:
    model: ModelConfig
    train: TrainConfig
    data: DataConfig


def default_config() -> Config:
    model_cfg = ModelConfig(
        codebook_sizes=[64, 32, 16],
        max_iter=10,
        nredo=1,
        max_points_per_centroid=256,
        spherical=False,
        random_state=42,
    )

    train_cfg = TrainConfig(
        output_dir="./rqkmeans_model",
    )

    data_cfg = DataConfig(
        embedding_dim=1024,
        train_samples=20480,
        test_samples=200,
    )

    return Config(
        model=model_cfg,
        train=train_cfg,
        data=data_cfg,
    )
