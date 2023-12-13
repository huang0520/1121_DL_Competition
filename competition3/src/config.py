from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

RANDOM_STATE: int = 42
AUTOTUNE = tf.data.experimental.AUTOTUNE
RNG_GENERATOR: np.random.Generator = np.random.default_rng(RANDOM_STATE)

UNCONDITIONAL_SENT = ["<|startoftext|>"]


@dataclass
class DirPath:
    data: Path = Path("./data")
    dict: Path = Path("./data/dictionary")
    dataset: Path = Path("./data/dataset")
    image: Path = Path("./data/102flowers")
    checkpoint: Path = Path("./checkpoints")
    output: Path = Path("./output")


@dataclass
class ModelConfig:
    image_size: int = 64
    max_seq_len: int = 20
    embedding_dim: int = 512
    widths: tuple[int] = (32, 64, 96, 128)
    attentions: tuple[bool] = (False, False, True, True)
    block_depth: int = 2
    embedding_max_freq: float = 1000.0
    start_log_snr: float = 3.0
    end_log_snr: float = -10.0
    kid_image_size: int = 75
    aug_prob: float = 0.3


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 50
    lr: float = 1e-4
    lr_init: float = 1e-5
    lr_decay: float = 1e-5
    ema: float = 0.999
    kid_diffusion_steps: int = 10
    plot_diffusion_steps: int = 40
    transfer: bool = False
