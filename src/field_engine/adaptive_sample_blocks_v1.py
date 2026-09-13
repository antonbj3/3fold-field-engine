"""Lossless adaptive spatial blocks with interpolation on the original fine grid.

The builder consumes a dense float32 field, but retains only flat node/payload
arrays. It collapses bitwise invariant axes and subdivides varying axes when
the complete retained byte cost decreases. Queries gather the eight fine-grid
samples across leaf boundaries before interpolation; there are no face writes.
"""
import itertools
import operator
from numbers import Real
import numpy as np
from scipy import ndimage

NODE = np.dtype([("lo", "<i4", (3,)), ("shape", "<i4", (3,)),
                 ("stored_shape", "<i4", (3,)), ("children", "<i4", (2,)),
                 ("offset", "<i8")])


def _build(array, lo, leaf_samples):
    selectors = []
    for axis in range(3):
        first = np.take(array, [0], axis=axis)
        invariant = np.all(array.view(np.uint32) == first.view(np.uint32))
        selectors.append(slice(0, 1) if invariant else slice(None))
    reduced = array[tuple(selectors)]
    direct_bytes = NODE.itemsize + reduced.nbytes
    leaf = (lo, array.shape, reduced, None, direct_bytes)
    if reduced.size <= 64 or array.size <= leaf_samples:
        return leaf
    axis = max((a for a in range(3) if reduced.shape[a] > 1), key=lambda a: array.shape[a])
    midpoint = array.shape[axis] // 2
    left_slice = [slice(None)] * 3
    right_slice = left_slice.copy()
    left_slice[axis] = slice(0, midpoint)
    right_slice[axis] = slice(midpoint, None)
    right_lo = list(lo)
    right_lo[axis] += midpoint
    left = _build(array[tuple(left_slice)], lo, leaf_samples)
    right = _build(array[tuple(right_slice)], tuple(right_lo), leaf_samples)
    split_bytes = NODE.itemsize + left[4] + right[4]
    if split_bytes < direct_bytes:
        return lo, array.shape, None, (left, right), split_bytes
    return leaf


def _flatten(node, records, pieces, offset):
    lo, shape, reduced, children, _ = node
    index = len(records)
    records.append(None)
    if children is None:
        records[index] = (lo, shape, reduced.shape, (-1, -1), offset)
        pieces.append(np.ascontiguousarray(reduced).ravel())
        offset += reduced.size
    else:
        left, offset = _flatten(children[0], records, pieces, offset)
        right, offset = _flatten(children[1], records, pieces, offset)
        records[index] = (lo, shape, (0, 0, 0), (left, right), -1)
    return index, offset


class AdaptiveSampleBlocks:
    def __init__(self, samples, origin, pitch, *, leaf_samples=512):
        samples = np.asarray(samples)
        if samples.dtype != np.float32 or samples.ndim != 3 or not samples.size:
            raise ValueError("Nonempty three-dimensional float32 samples required")
        if max(samples.shape) > 4096 or samples.size > 16777216 or not np.isfinite(samples).all():
            raise ValueError("Finite bounded field required")
        origin = np.asarray(origin, dtype=np.float64)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("Finite three-coordinate origin required")
        if isinstance(pitch, (bool, np.bool_)) or not isinstance(pitch, Real):
            raise ValueError("Positive finite scalar pitch required")
        try:
            pitch = float(pitch)
        except (ValueError, TypeError):
            raise ValueError("Positive finite scalar pitch required") from None
        if not np.isfinite(pitch) or pitch <= 0:
            raise ValueError("Positive finite scalar pitch required")
        if isinstance(leaf_samples, (bool, np.bool_)):
            raise ValueError("Integer leaf size required")
        try:
            leaf_samples = operator.index(leaf_samples)
        except TypeError:
            raise ValueError("Integer leaf size required") from None
        if not 8 <= leaf_samples <= 4096:
            raise ValueError("Leaf size outside bounded contract")
        root = _build(samples, (0, 0, 0), leaf_samples)
        records, pieces = [], []
        _flatten(root, records, pieces, 0)
        self.nodes = np.array(records, dtype=NODE)
        self.payload = np.concatenate(pieces)
        self.origin = origin.copy()
        self.pitch = pitch
        self.shape = samples.shape
        for array in (self.nodes, self.payload, self.origin):
            array.flags.writeable = False

    @property
    def storage_bytes(self):
        # Include origin, pitch, and dimensions as well as every node and sample.
        return self.nodes.nbytes + self.payload.nbytes + self.origin.nbytes + 8 + 3 * 8

    def samples_at(self, indices):
        indices = np.asarray(indices)
        if indices.ndim != 2 or indices.shape[1] != 3 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("Integer index triples required")
        if np.any(indices < 0) or np.any(indices >= np.array(self.shape)):
            raise ValueError("Sample index outside field")
        out = np.empty(len(indices), dtype=np.float32)
        pending = [(0, np.arange(len(indices)))]
        while pending:
            node_id, selected = pending.pop()
            if not len(selected):
                continue
            node = self.nodes[node_id]
            left, right = node["children"]
            if left < 0:
                local = indices[selected] - node["lo"]
                local[:, node["stored_shape"] == 1] = 0
                linear = np.ravel_multi_index(local.T, tuple(node["stored_shape"]))
                out[selected] = self.payload[node["offset"] + linear]
            else:
                bounds = self.nodes[left]
                in_left = np.all((indices[selected] >= bounds["lo"]) &
                                 (indices[selected] < bounds["lo"] + bounds["shape"]), axis=1)
                pending.append((int(right), selected[~in_left]))
                pending.append((int(left), selected[in_left]))
        return out

    def query(self, points):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Finite point triples required")
        out = np.empty(len(points), dtype=np.float32)
        offsets = np.array(list(itertools.product((0, 1), repeat=3)), dtype=np.int64)
        for start in range(0, len(points), 4096):
            batch = points[start:start + 4096]
            with np.errstate(over="ignore", invalid="ignore"):
                coordinates = (batch - self.origin) / self.pitch
            if not np.isfinite(coordinates).all():
                raise ValueError("Query exceeds finite grid-coordinate range")
            coordinates = np.clip(coordinates, 0, np.array(self.shape) - 1)
            lower = np.floor(coordinates).astype(np.int64)
            fraction = coordinates - lower
            neighbors = np.minimum(lower[:, None, :] + offsets, np.array(self.shape) - 1)
            corners = self.samples_at(neighbors.reshape(-1, 3)).reshape(-1, 2, 2, 2)
            # The leading coordinate is exactly integral, selecting one eight-sample
            # cell. The remaining axes use the original fine-grid fractions.
            query = np.vstack((np.arange(len(batch)), fraction.T))
            out[start:start + len(batch)] = ndimage.map_coordinates(corners, query, order=1, mode="nearest")
        return out


if __name__ == "__main__":
    import sys as _probe_sys
    from pathlib import Path as _ProbePath
    _probe_sys.path.insert(0, str(_ProbePath(__file__).resolve().parents[2] / 'probes/field_engine'))
    from field_adaptive_blocks_probe import main
    raise SystemExit(main())
