from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
import toml

RANDOM_STATE: int = 42
AUTOTUNE = tf.data.experimental.AUTOTUNE
RNG_GENERATOR: np.random.Generator = np.random.default_rng(RANDOM_STATE)


@dataclass
class DirPath:
    data: Path = Path("./data")
    dict: Path = Path("./data/dictionary")
    dataset: Path = Path("./data/dataset")
    image: Path = Path("./data/102flowers")
    checkpoint: Path = Path("./checkpoints")
    output: Path = Path("./output")
    log: Path = Path("./logs")


@dataclass
class DatasetConfig:
    aug_prob: float = 0.4
    max_seq_len: int = 20


@dataclass
class ModelConfig:
    image_size: int = 64
    noise_embedding_dim: int = 512
    image_embedding_dim: int = 64
    text_embedding_shape: tuple[int] = (DatasetConfig.max_seq_len, 512)
    widths: tuple[int] = (64, 96, 128, 160)
    block_depth: int = 2
    embedding_max_frequency: float = 1000.0
    max_signal_rate: float = 0.95
    min_signal_rate: float = 0.02


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-3
    lr_init: float = 3e-5
    lr_decay: float = 6e-4
    ema: float = 0.999
    plot_diffusion_steps: int = 100
    transfer: bool = False


def export_config(path: Path):
    output_dict = {
        "DatasetConfig": asdict(DatasetConfig()),
        "ModelConfig": asdict(ModelConfig()),
        "TrainConfig": asdict(TrainConfig()),
    }

    with path.open("w") as f_write:
        toml.dump(output_dict, f_write)
