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

    base_name = os.path.basename(json_path).split('.')[0]

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

def _save_grid(grid, out_path):
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0

    # Flatten and convert to float32
    flat_data = np.array(grid, dtype=np.float32).flatten()

    with open(out_path, 'wb') as f:
        # Write header: two 64-bit signed integers (little-endian)
        f.write(struct.pack('<qq', rows, cols))
        # Write payload
        f.write(flat_data.tobytes())

# Example usage (commented out to allow import)
# if __name__ == "__main__":
#     compile_arc_json('data/training/007bbfb7.json', 'data_bin')
