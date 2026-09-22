from . import DataModuleConfig
from omegaconf import MISSING
from dataclasses import dataclass
from typing import Optional

@dataclass
class NPYDataModuleConfig(DataModuleConfig):
    _target_: str = "forgetting_transformer.datamodule.npy.NpyDataModule"
    data_path: str = "${data_dir}"
    rank: int = MISSING
    world_size: int = MISSING
    train_batch_len: int = MISSING
    train_batch_size: int = MISSING
    train_num_workers: int = MISSING
    eval_tokens: int = MISSING
    eval_batch_len: int = MISSING
    eval_local_batch_size: int = MISSING
    eval_num_workers: int = MISSING