# Round-trips an ARC grid through the compiled .bin format and checks that a
# malformed (truncated) file is rejected instead of read out of bounds.
# run_tests.sh generates build/sample_in.bin from arc_compiler.py first.
from arc_io import load_arc_grid

comptime SAMPLE_PATH = "build/sample_in.bin"
comptime BAD_PATH = "build/malformed.bin"


def main() raises:
    # Valid file: arc_compiler.py wrote a 2x3 grid [[1,2,3],[4,5,6]].
    var grid = load_arc_grid(SAMPLE_PATH)
    if grid.rows != 2 or grid.cols != 3:
        raise Error("ERROR: loaded grid has wrong dimensions.")
    if grid.get(0, 0) != 1.0 or grid.get(1, 2) != 6.0:
        raise Error("ERROR: loaded grid has wrong cell values.")

    # Malformed file: header claims a payload longer than the file. Writing a
    # 4-byte file (smaller than the 16-byte header) must raise.
    var f = open(BAD_PATH, "w")
    f.write("xx")
    f.close()
    var rejected = False
    try:
        var bad = load_arc_grid(BAD_PATH)
    except e:
        rejected = True
    if not rejected:
        raise Error("ERROR: malformed .bin was not rejected.")

    print("ARC IO tests passed.")
