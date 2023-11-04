import zarr
import torch
import numpy as np
from torch.utils.data import IterableDataset
from typing import Iterator, List, Optional

class VesuviusStream(IterableDataset):
    file_path: List[str]
    z_size: int
    y_size: int
    x_size: int
    samples_per_epoch: int
    current_sample: int
    s: List[zarr.Array]
    file_probabilities: Optional[List[float]]
    samples_per_file: List[int]
    sampling_method: str
    shuffle: bool
    total_samples: int

    def __init__(self, file_paths: List[str], z_size: int, y_size: int, x_size: int, samples_per_epoch: int,
                 sampling_method: str = 'uniform', shuffle: bool = True):
        assert len(file_paths), "No file in the file paths list."
        assert sampling_method in ['uniform', 'proportional'], "sampling_method must be 'uniform' or 'proportional'"

        self.file_paths = file_paths
        self.z_size = z_size
        self.y_size = y_size
        self.x_size = x_size
        self.samples_per_epoch = samples_per_epoch
        self.current_sample = 0
        self.sampling_method = sampling_method
        self.shuffle = shuffle
        # Open all Zarr files and store them in a list
        self.s_list = [zarr.open(file_path, mode='r') for file_path in file_paths]

        self.z_range = [s.shape[0] - z_size for s in self.s_list]
        self.y_range = [s.shape[1] - y_size for s in self.s_list]
        self.x_range = [s.shape[2] - x_size for s in self.s_list]
        
        assert all(z >= 0 for z in self.z_range), "z_size greater than max z dimension for some file"
        assert all(y >= 0 for y in self.y_range), "y_size greater than max y dimension for some file"
        assert all(x >= 0 for x in self.x_range), "x_size greater than max x dimension for some file"

        # Calculate the number of samples per file based on the dimension ranges
        self.calculate_samples_info()

        # Initialize the coordinate generators per file
        self.coordinate_generators = [
            self._coordinate_generator(self.z_range[i], self.y_range[i], self.x_range[i], self.shuffle)
            for i in range(len(self.file_paths))
        ]
    
    @staticmethod
    def _coordinate_generator(z_range: int, y_range: int, x_range: int, shuffle: bool):
      total_coords = z_range * y_range * x_range
      indices = np.arange(total_coords)
      if shuffle:
        np.random.shuffle(indices)
      for idx in indices:
        z = idx // (y_range * x_range)
        y = (idx // x_range) % y_range
        x = idx % x_range
        yield (z, y, x)
    
    def calculate_samples_info(self):
        self.samples_per_file = [z * y * x for z, y, x in zip(self.z_range, self.y_range, self.x_range)]
        self.total_samples = sum(self.samples_per_file)
        if self.sampling_method == 'proportional':
            self.file_probabilities = [samples / self.total_samples for samples in self.samples_per_file]
        else:
            self.file_probabilities = None

    def update_probabilities(self, idx):
        # Recalculate the number of samples per file and total samples
        self.samples_per_file[idx] -= 1
        total_samples = sum(self.samples_per_file)
        if self.sampling_method == 'proportional':
            self.file_probabilities = [samples / total_samples for samples in self.samples_per_file]
        else:
            self.file_probabilities = None

    def __iter__(self) -> Iterator['VesuviusStream']:
        self.current_sample = 0
        self.calculate_samples_info()
        # Initialize the coordinate generators per file
        self.coordinate_generators = [
            self._coordinate_generator(self.z_range[i], self.y_range[i], self.x_range[i], self.shuffle)
            for i in range(len(self.file_paths))
        ]

        return self

    def __len__(self) -> int:
        return self.total_samples
    
    def __next__(self) -> torch.Tensor:
        if self.current_sample >= self.samples_per_epoch:
          raise StopIteration

        # Select a file based on the specified sampling method
        file_idx = np.random.choice(
            len(self.file_paths),
            p=self.file_probabilities if self.sampling_method == 'proportional' else None
        )

        # Fetch coordinates from the generator
        try:
            z_start, y_start, x_start = next(self.coordinate_generators[file_idx])
        except StopIteration:
            # No remaining coordinates to sample from in file index
            return self.__next__()  # Skip to next file or end iteration

        block = self.fetch_block(self.s_list[file_idx], z_start, self.z_size, y_start, self.y_size, x_start, self.x_size)
        self.current_sample += 1

        # Update probabilities after sampling
        self.update_probabilities(file_idx)

        ### INSERT YOUR TRANSFORMATIONS AND AUGMENTATIONS HERE :)

        return torch.from_numpy(block).to(torch.int64)
    
    @staticmethod
    def fetch_block(s: zarr.Array, z_start: int, z_size: int, y_start: int, y_size: int, x_start: int, x_size: int) -> np.ndarray:
        # Fetch a single block from the dataset
        block = np.empty((z_size, y_size, x_size), dtype=np.int32)
        block = s[z_start:z_start + z_size, y_start:y_start + y_size, x_start:x_start + x_size]
        return block
    

