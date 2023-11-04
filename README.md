# Vesuvius 3D DataStreamer

## Introduction

The Vesuvius 3D DataStreamer is crafted to meet the growing demands of processing an increasing number of high-resolution 3D segments from the Herculaneum scrolls of the Vesuvius Challenge (https://scrollprize.org/).

It is designed to bypass the limitations of in-memory data loading by directly streaming from disk sampled 3D blocks, which becomes increasingly critical in training pipelines as the dataset expands. 

## Components
### datautils
`streamer.py`: Implements `VesuviusStream`, a PyTorch `IterableDataset` for streaming 3D chunks from multiple Zarr archives. It is designed for efficiency, only loading the necessary data for each chunk. Can sample with two strategies: `uniform` will first uniformly select one file, and then sample without replacement within the file; `proportional` will select one file with a probability proportional to the number of samples in it (more samples, more probability).

### tools
`converter.py`: A script to convert TIFF files into a Zarr archive, with options for cropping and chunking, facilitating efficient 3D array manipulation.

### Example Data
`example_zarr` folder contains a sample `example.zarr` archive, created from the "monster scroll" with specified 3D region of interest parameters.

### Training Example
`training_example.ipynb`: Demonstrates using `VesuviusStream` with PyTorch's `DataLoader` for a machine learning model.

## Installation
Install the required packages:
```console
pip install -r requirements.txt
```


## Usage

### Converting TIFF to Zarr
First one has to convert the TIFF images to Zarr for efficient access:
```console
python tools/converter.py /path/to/tiff_folder/ /path/to/destination.zarr --parameters
```
All TIFF files from `00.tif` to `64.tif` for a segment or scroll must be present in the folder.

The parameters can be set to specify the ROI (region of interest) in the 3D images in terms of 3D coordinates.

`example.zarr` has been generated with the following (ROI) parameters:
 - --z_start 26 --z_end 36
 - --x_start 6000 --x_end 7000
 - --y_start 4000 --y_end 5000

Zarr saves multidimensional arrays in separated chunks.
The chunk size parameters here determine the shapes that zarr will use to split the data, and not the 3D chunks used for the ML model.
By default, chunks are:
 - --z_chunksize 4
 - --y_chunksize 512
 - --y_chunksize 512

This setting will produce ~2MB chunks, but feel free to play with the settings.

### Data Streaming
VesuviusStream loads data on-the-fly:
```python
from datautils.streamer import VesuviusStream
from torch.utils.data import DataLoader

dataset = VesuviusStream(file_paths=['./example_zarr/example.zarr'], z_size=2, y_size=4, x_size=6, samples_per_epoch=16, sampling_method='uniform', shuffle=True)
loader = DataLoader(dataset, batch_size=4, num_workers=2)
```

Here the parameters `z_size`, `y_size` and `x_size` indicate the shape of the 3D block to sample from the scroll/fragment.

### Training Example
Refer to training_example.ipynb for integrating data streaming into a training loop.

### Memory Efficience
The streamer reads only the chunks in the Zarr archive necessary to produce the block. This is why it is memory efficient.
It is recommended to set the chunksize parameters in the converter greater than the dimensions of the blocks to fetch.

## Framework Compatibility
The tool is built for PyTorch but can be adapted for other frameworks.

## Contributing
Contributions are welcome.

## Author
Dr. Giorgio Angelotti

For any inquiries or further information, feel free to contact me at giorgio.angelotti@isae-supaero.fr

## License
This project is licensed under the MIT License -- see the `LICENSE` file for details.




