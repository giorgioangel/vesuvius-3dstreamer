import zarr
import torch
import numpy as np
from torch.utils.data import IterableDataset
from typing import Iterator

class VesuviusStream(IterableDataset):
    file_path: str
    z_size: int
    row_size: int
    col_size: int
    samples_per_epoch: int
    current_sample: int
    s: zarr.Array

    def __init__(self, file_path: str, z_size: int, y_size: int, x_size: int, samples_per_epoch: int):
        self.file_path = file_path
        self.z_size = z_size
        self.y_size = y_size
        self.x_size = x_size
        self.s = zarr.open(self.file_path, mode='r')
        self.samples_per_epoch = samples_per_epoch
        self.current_sample = 0

    def __iter__(self) -> Iterator['VesuviusStream']:
        return self

    def __next__(self) -> torch.Tensor:
        if self.current_sample >= self.samples_per_epoch:
          raise StopIteration
        z_start = np.random.choice(self.s.shape[0] - self.z_size)
        y_start = np.random.choice(self.s.shape[1] - self.y_size)
        x_start = np.random.choice(self.s.shape[2] - self.x_size)
        block = self.fetch_block(self.s, z_start, self.z_size, y_start, self.y_size, x_start, self.x_size)
        self.current_sample += 1

        ### INSERT YOUR TRANSFORMATIONS AND AUGMENTATIONS HERE :)

        return torch.from_numpy(block).to(torch.int64)
    
    @staticmethod
    def fetch_block(s: zarr.Array, z_start: int, z_size: int, y_start: int, y_size: int, x_start: int, x_size: int) -> np.ndarray:
        # Fetch a single block from the dataset
        block = np.empty((z_size, y_size, x_size), dtype=np.int32)
        block = s[z_start:z_start + z_size, y_start:y_start + y_size, x_start:x_start + x_size]
        return block.astype(np.int32)
    

