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
class ModelConfig:
    image_size: int = 64
    max_seq_len: int = 20
    embedding_dim: int = 512
    widths: tuple[int] = (64, 128, 256, 512)
    block_depth: int = 2
    embedding_max_freq: float = 1000.0
    start_log_snr: float = 3.0
    end_log_snr: float = -10.0
    kid_image_size: int = 75
    aug_prob: float = 0.4


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    lr_init: float = 1e-4
    lr_decay: float = 1e-3
    ema: float = 0.999
    kid_diffusion_steps: int = 10
    plot_diffusion_steps: int = 50
    transfer: bool = False
    cfg_scale: float = 7.5


def export_config(path: Path):
    output_dict = {
        "ModelConfig": asdict(ModelConfig()),
        "TrainConfig": asdict(TrainConfig()),
    }

    with path.open("w") as f_write:
        toml.dump(output_dict, f_write)
