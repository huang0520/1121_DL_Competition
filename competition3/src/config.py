from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
import toml

RANDOM_STATE: int = 0
AUTOTUNE = tf.data.experimental.AUTOTUNE
RNG_GENERATOR: np.random.Generator = np.random.default_rng(RANDOM_STATE)


@dataclass
class DirPath:
    data: Path = Path("./data")
    dict: Path = Path("./data/dictionary")
    dataset: Path = Path("./data/dataset")
    original_image: Path = Path("./data/102flowers")
    resize_image: Path = Path("./data/resize_image")
    checkpoint: Path = Path("./checkpoints")
    output: Path = Path("./output")
    log: Path = Path("./logs")


@dataclass
class DatasetConfig:
    aug_prob: float = 0.3
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
    start_log_snr: float = 2.5
    end_log_snr: float = -7.5


@dataclass
class TrainConfig:
    batch_size: int = 48
    epochs: int = 50
    lr: float = 1e-4
    lr_init: float = 5e-5
    weight_decay: float = 5e-4
    ema: float = 0.999
    plot_diffusion_steps: int = 100
    transfer: bool = True
    cfg_scale: float = 3.6


def export_config(path: Path):
    output_dict = {
        "DatasetConfig": asdict(DatasetConfig()),
        "ModelConfig": asdict(ModelConfig()),
        "TrainConfig": asdict(TrainConfig()),
    }

    with path.open("w") as f_write:
        toml.dump(output_dict, f_write)
