"""Compress the common CPU/CUDA/RT mesh-stage result without changing its field."""
from dataclasses import dataclass
import numpy as np
from adaptive_sample_blocks_v1 import AdaptiveSampleBlocks


@dataclass(frozen=True)
class AdaptiveMeshField:
    samples: AdaptiveSampleBlocks
    gmin: np.ndarray

    @property
    def shape(self):
        return self.samples.shape

    @property
    def storage_bytes(self):
        return self.samples.storage_bytes + self.gmin.nbytes

    def query(self, points):
        return self.samples.query(points)


def mesh_to_adaptive(stage, vertices, faces, pitch, origin, **stage_kwargs):
    """Run an explicit mesh stage and retain only its lossless adaptive field.

    ``stage`` returns the frozen (gmin, shape, surface, solid, distance) tuple.
    Backend choice, native library paths and sign policy stay with that stage.
    Dense masks/distances exist transiently during construction; no peak-memory
    reduction is implied by the smaller returned representation.
    """
    if not callable(stage):
        raise ValueError('An explicit callable mesh stage is required')
    result = stage(vertices, faces, pitch, origin, **stage_kwargs)
    if len(result) != 5:
        raise ValueError('Expected the five-part mesh-stage result')
    gmin = np.asarray(result[0])
    if gmin.shape != (3,) or not np.issubdtype(gmin.dtype, np.integer):
        raise ValueError('Integral three-coordinate grid offset required')
    distance = np.asarray(result[4])
    if tuple(result[1]) != distance.shape:
        raise ValueError('Stage shape disagrees with distance field')
    anchor = np.asarray(origin, dtype=np.float64) + gmin * pitch
    blocks = AdaptiveSampleBlocks(distance, anchor, pitch)
    offset = gmin.copy()
    offset.flags.writeable = False
    return AdaptiveMeshField(blocks, offset)


if __name__ == '__main__':
    import sys as _probe_sys
    from pathlib import Path as _ProbePath
    _probe_sys.path.insert(0, str(_ProbePath(__file__).resolve().parents[2] / 'probes/field_engine'))
    from field_mesh_adaptive_probe import main
    raise SystemExit(main())
