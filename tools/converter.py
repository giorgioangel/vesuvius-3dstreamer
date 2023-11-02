import zarr
import tifffile
import gc
import numpy as np
import os
from typing import Optional

def list_files(directory: str, extension: str) -> list:
    return [directory+f for f in os.listdir(directory) if f.endswith(extension)]

def get_dimensions(file:str) -> (int, int):
    memmap_file = None
    try:
        memmap_file = tifffile.memmap(file)
        dimensions = memmap_file.shape
        return dimensions
    except Exception as e:
        print(f"Error reading TIFF files: {e}")
        return -1, -1
    finally:
        # Clear memory
        if memmap_file is not None:
            del memmap_file
            gc.collect()


def saveaszarr(tiff_folder: str, destination: str, y_start: Optional[int] = None, y_end: Optional[int] = None,
                x_start: Optional[int] = None, x_end: Optional[int] = None,
                z_start: Optional[int] = None, z_end: Optional[int] = None,
                z_chunksize: int = 4, y_chunksize: int = 512, x_chunksize: int = 512):
    if (y_start is not None) and (y_end is not None):
        assert y_end > y_start, "y_end should be greater than y_start"

    if (x_start is not None) and (x_end is not None):
        assert x_end > x_start, "x_end should be greater than x_start"
    
    if (z_start is not None) and (z_end is not None):
        assert z_end > z_start, "z_end should be greater than z_start"

    if (z_start is not None):
        assert 0 <= z_start <= 64, "Invalid value for z_start"
    else:
        z_start = 0
    
    if (z_end is not None):
        assert 0 <= z_end <= 64, "Invalid value for z_end"
    else:
        z_end = 64
    
    
    paths = list_files(tiff_folder, '.tif')

    assert len(paths), "The folder does not contain any tiff files"

    ydim, xdim = get_dimensions(paths[0])

    assert ydim != -1, "Could not get the 2D dimensions"

    if (y_start is not None):
        assert 0 <= y_start <= ydim, "Invalid value for y_start"
    else:
        y_start = 0
    
    if (y_end is not None):
        assert 0 <= y_end <= ydim, "Invalid value for y_end"
    else:
        y_end = ydim

    if (x_start is not None):
        assert 0 <= x_start <= xdim, "Invalid value for x_start"
    else:
        x_start = 0
    
    if (x_end is not None):
        assert 0 <= x_end <= xdim, "Invalid value for x_end"
    else:
        x_end = xdim

    assert 1 <= z_chunksize <= z_end - z_start, "Invalid z chunk size"
    assert 1 <= y_chunksize <= y_end - y_start, "Invalid y chunk size"
    assert 1 <= x_chunksize <= x_end - x_start, "Invalid x chunk size"

    # Create a memory-mapped array
    memmap_files = []
    try:
        for i in range(z_start, z_end+1):
            memmap_file = tifffile.memmap(paths[i])
            memmap_files.append(memmap_file)
    except Exception as e:
        print(f"Error reading TIFF files: {e}")
        return

    try:
        # Stack the memory-mapped arrays along a new dimension
        stacked_array = np.stack(memmap_files, axis=0)

        # Cropping in 2D if needed
        stacked_array = stacked_array[:,y_start:y_end,x_start:x_end]
        
        z = zarr.open(destination+'.zarr', mode='w', shape=stacked_array.shape, dtype=stacked_array.dtype, chunks=(z_chunksize,y_chunksize,x_chunksize))

        z[:] = stacked_array
    except Exception as e:
        print(f"Error writing Zarr file: {e}") 
    finally:
        # Clearing memory
        for memmap_file in memmap_files:
            del memmap_file
        gc.collect()

# For the command-line interface
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Convert TIFF files to Zarr format.")
    parser.add_argument('folder', type=str, help='Path to the TIFF files folder to convert')
    parser.add_argument('destination', type=str, help='Destination Zarr file path.')
    parser.add_argument('--y_start', type=int, help='Start index for Y dimension cropping.')
    parser.add_argument('--y_end', type=int, help='End index for Y dimension cropping.')
    parser.add_argument('--x_start', type=int, help='Start index for X dimension cropping.')
    parser.add_argument('--x_end', type=int, help='End index for X dimension cropping.')
    parser.add_argument('--z_start', type=int, help='Start index for Z dimension.')
    parser.add_argument('--z_end', type=int, help='End index for Z dimension.')
    parser.add_argument('--z_chunksize', type=int, default=4, help='Chunk size for Z dimension in Zarr file.')
    parser.add_argument('--y_chunksize', type=int, default=512, help='Chunk size for Y dimension in Zarr file.')
    parser.add_argument('--x_chunksize', type=int, default=512, help='Chunk size for X dimension in Zarr file.')

    args = parser.parse_args()

    # Call the function with the parsed arguments
    saveaszarr(tiff_folder=args.folder, destination=args.destination,
               y_start=args.y_start, y_end=args.y_end,
               x_start=args.x_start, x_end=args.x_end,
               z_start=args.z_start, z_end=args.z_end,
               z_chunksize=args.z_chunksize, y_chunksize=args.y_chunksize, x_chunksize=args.x_chunksize)







