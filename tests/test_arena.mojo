from memory import UnsafePointer, memset_zero
from sys import sizeof

struct HopeArena:
    var data: UnsafePointer[UInt8]
    var capacity: Int
    var offset: Int

    fn __init__(inout self, capacity: Int):
        self.capacity = capacity
        self.offset = 0
        self.data = UnsafePointer[UInt8].alloc(self.capacity)
        memset_zero(self.data, self.capacity)

    fn __moveinit__(inout self, owned existing: Self):
        self.data = existing.data
        self.capacity = existing.capacity
        self.offset = existing.offset

        existing.data = UnsafePointer[UInt8]()
        existing.capacity = 0
        existing.offset = 0

    fn __del__(owned self):
        if self.data:
            self.data.free()

    fn alloc_node[T: AnyType](inout self) -> UnsafePointer[T]:
        var size = sizeof[T]()
        if self.offset + size > self.capacity:
            return UnsafePointer[T]()

        var ptr = self.data.offset(self.offset).bitcast[T]()
        self.offset += size
        return ptr

fn main() raises:
    var arena = HopeArena(1024)
    var float_ptr = arena.alloc_node[Float32]()

    # Assert pointer is valid (not null)
    if not float_ptr:
        raise Error("ERROR: Arena failed to allocate memory or returned null.")

    float_ptr.store(0, 42.0)
    var val = float_ptr.load(0)

    if val != 42.0:
        raise Error("ERROR: Memory read/write failure.")

    print("HopeArena basic allocation test passed.")
