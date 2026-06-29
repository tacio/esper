import json
import struct
import numpy as np
import os

def compile_arc_json(json_path: str, output_dir: str):
    """
    Parses an ARC JSON file and converts each input/output grid into a raw .bin file.
    The binary format consists of two 64-bit integers (rows, cols) followed by
    the flattened grid as 32-bit floats.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Use splitext so task ids containing dots are not truncated.
    base_name = os.path.splitext(os.path.basename(json_path))[0]

    # Process training pairs
    if 'train' in data:
        for i, pair in enumerate(data['train']):
            _save_grid(pair['input'], f"{output_dir}/{base_name}_train_{i}_in.bin")
            _save_grid(pair['output'], f"{output_dir}/{base_name}_train_{i}_out.bin")

    # Process test pairs
    if 'test' in data:
        for i, pair in enumerate(data['test']):
            _save_grid(pair['input'], f"{output_dir}/{base_name}_test_{i}_in.bin")
            if 'output' in pair:
                _save_grid(pair['output'], f"{output_dir}/{base_name}_test_{i}_out.bin")

def _write_grid(f, grid):
    """Write one grid-block to an open binary file: two little-endian int64
    (rows, cols) followed by the flattened grid as float32. This is the single
    grid encoding shared by the per-grid `.bin` format and the `.task` bundle."""
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    flat_data = np.array(grid, dtype=np.float32).flatten()
    f.write(struct.pack('<qq', rows, cols))
    f.write(flat_data.tobytes())


def _save_grid(grid, out_path):
    with open(out_path, 'wb') as f:
        _write_grid(f, grid)


def _save_task(train_pairs, test_pairs, out_path):
    """Write a whole task as one binary bundle (a `.task` file) the Mojo engine
    loads in a single read (see `load_arc_task` in `arc_io.mojo`):

        [n_train: int64][n_test: int64]
        then 2*n_train + 2*n_test grid-blocks in order:
        train_in, train_out, ..., test_in, test_out

    Each (input, output) pair is a 2-tuple of 2-D list-of-lists grids. Bundling
    avoids Mojo-side directory globbing and text parsing — the shell globs the
    `.task` files into the driver's argv instead.
    """
    with open(out_path, 'wb') as f:
        f.write(struct.pack('<qq', len(train_pairs), len(test_pairs)))
        for grid_in, grid_out in train_pairs:
            _write_grid(f, grid_in)
            _write_grid(f, grid_out)
        for grid_in, grid_out in test_pairs:
            _write_grid(f, grid_in)
            _write_grid(f, grid_out)

# Example usage (commented out to allow import)
# if __name__ == "__main__":
#     compile_arc_json('data/training/007bbfb7.json', 'data_bin')
