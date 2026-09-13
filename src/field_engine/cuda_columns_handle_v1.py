"""Explicit CUDA scan backend through the frozen validated compact column seam."""
from rt_columns_compact_handle_v1 import ColumnsHandle as CompactColumnsHandle


class ColumnsHandle(CompactColumnsHandle):
    def __init__(self, library, corners, anchor, capacity, hit_capacity=64):
        super().__init__(library, "cuda-scan-v1", corners, anchor, capacity, hit_capacity)
