import numpy as np
import torch
from pathlib import Path
from typing import Optional, Union, Literal
import lightning as L
from torch.utils.data import Dataset, DataLoader, Sampler

from forgetting_transformer.datamodule.common import DataInfo, Batch
from forgetting_transformer.utils import safe_divide, check_divisible

class NpyDataset(Dataset):
    def __init__(
        self,
        data_path: Union[str, Path],
        split: Literal["train", "heldout"],
        world_size: int,
        rank: int,
        batch_size: int,
        batch_len: int,
        total_tokens: Optional[int] = None,
        bos_token_id: Optional[int] = 50256,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.data = np.load(self.data_path, mmap_mode="r")
        self.num_rows, self.row_len = self.data.shape
        self.split = split
        if self.row_len == batch_len + 1:
            pass
        elif self.row_len > batch_len + 1:
            self.data = self.data[:, :batch_len + 1]
            self.row_len = batch_len + 1
        elif self.row_len < batch_len + 1:
            raise ValueError(f"Data row_len={self.row_len} is smaller than batch_len={batch_len}")

        assert self.row_len in (batch_len, batch_len + 1), \
            f"batch_len={batch_len} which is different from row_len={self.row_len}!"
        self.batch_len = batch_len


        self.world_size = world_size
        self.rank = rank
        self.global_batch_size = batch_size
        self.local_batch_size = safe_divide(self.global_batch_size, world_size)


        check_divisible(self.num_rows, self.global_batch_size)
        self.batch_count = safe_divide(self.num_rows, self.global_batch_size)

        self.tokens_per_batch = self.global_batch_size * self.batch_len
        self.local_tokens_per_batch = self.local_batch_size * self.batch_len

        if total_tokens is None:
            self.total_tokens = self.batch_count * self.tokens_per_batch
        else:
            check_divisible(total_tokens, self.tokens_per_batch)
            keep_batches = safe_divide(total_tokens, self.tokens_per_batch)
            self.batch_count = keep_batches
            self.total_tokens = self.batch_count * self.tokens_per_batch

        self.bos_token_id = bos_token_id

    def __len__(self):
        return self.batch_count

    def __getitem__(self, batch_id: int) -> Batch:
        assert 0 <= batch_id < self.batch_count
        global_row_start = batch_id * self.global_batch_size

        start_row = global_row_start + self.rank * self.local_batch_size
        end_row   = start_row + self.local_batch_size


        
        data = self.data[start_row:end_row, :self.batch_len + 1]   
        if data.shape[0] == 0:
            print(f"[WARN] Skipping empty batch_id={batch_id}", flush=True)
            raise StopIteration  


        assert data.shape[1] == self.batch_len + 1, \
            f"Expected {self.batch_len + 1} tokens per row, got {data.shape[1]}"

        labels = np.array(data[:, 1:], dtype=np.int64)
        input_ids = np.array(data[:,:-1], dtype=np.int64)

        
        if self.bos_token_id is not None:
            input_ids[..., 0] = self.bos_token_id

        resets = np.zeros_like(labels, dtype=np.bool_)
        resets[..., 0] = True

        return Batch(input_ids=input_ids, labels=labels, resets=resets)


class StatefulSampler(Sampler):
    def __init__(self, dataset: Dataset):
        super().__init__()
        self.dataset = dataset
        self.batch_id = 0

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        while self.batch_id < len(self):
            cur = self.batch_id
            self.batch_id += 1
            yield cur

    def state_dict(self):
        return {"batch_id": self.batch_id}

    def load_state_dict(self, state_dict):
        self.batch_id = state_dict["batch_id"]


class NpyDataloader(DataLoader):
    def state_dict(self):
        assert isinstance(self.sampler, StatefulSampler)
        return self.sampler.state_dict()

    def load_state_dict(self, state_dict):
        assert isinstance(self.sampler, StatefulSampler)
        self.sampler.load_state_dict(state_dict)


class NpyDataModule(L.LightningDataModule):

    def __init__(
        self,
        data_path,
        world_size: int,
        rank: int,
        train_batch_len: int,
        train_batch_size: int,
        train_num_workers: int,
        eval_batch_len: int,
        eval_local_batch_size: int,
        eval_num_workers: int,
        eval_tokens: Optional[int] = None,
        bos_token_id: Optional[int] = 0,
        vocab_size: int = 50277,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.world_size = world_size
        self.rank = rank
        self.train_num_workers = train_num_workers
        self.eval_num_workers = eval_num_workers
        self.data_dir = Path(data_path)
        self.train_dataset = NpyDataset(
            data_path=self.data_dir / "train.npy",
            split="train",
            world_size=world_size,
            rank=rank,
            batch_size=train_batch_size,
            batch_len=train_batch_len,
            total_tokens=None,
            bos_token_id=bos_token_id,
        )

        self.val_dataset = NpyDataset(
            data_path=self.data_dir / "eval.npy",
            split="heldout",
            world_size=world_size,
            rank=rank,
            batch_size=eval_local_batch_size * world_size,
            batch_len=eval_batch_len,
            total_tokens=eval_tokens,
            bos_token_id=bos_token_id,
        )

    def train_dataloader(self):
        sampler = StatefulSampler(self.train_dataset)
        dl = NpyDataloader(
            self.train_dataset,
            batch_size=None,
            shuffle=False,
            sampler=sampler,
            num_workers=self.train_num_workers,
            pin_memory=True,
        )
        info = DataInfo(
            vocab_size=self.vocab_size,
            batch_len=self.train_dataset.batch_len,
            global_tokens_per_batch=self.train_dataset.tokens_per_batch,
            local_tokens_per_batch=self.train_dataset.local_tokens_per_batch,
            seq_len=self.train_dataset.batch_len,
            total_tokens=self.train_dataset.total_tokens,
        )
        return dl, info

    def val_dataloader(self):
        dl = NpyDataloader(
            self.val_dataset,
            batch_size=None,
            shuffle=False,
            sampler=None,
            num_workers=self.eval_num_workers,
            pin_memory=True,
        )
        info = DataInfo(
            vocab_size=self.vocab_size,
            batch_len=self.val_dataset.batch_len,
            global_tokens_per_batch=self.val_dataset.tokens_per_batch,
            local_tokens_per_batch=self.val_dataset.local_tokens_per_batch,
            seq_len=self.val_dataset.batch_len,
            total_tokens=self.val_dataset.total_tokens,
        )
        return dl, info

