"""Subcell wall links from the engine's sampled SDF, in any consistent length unit.

No geometry is evaluated here. The zero of the linear SDF along each lattice link gives its
fraction theta. This retains subcell information when the SDF has it; a signed EDT of a binary
mask cannot reconstruct the original subcell boundary. Directions are integer lattice vectors.
"""
import numpy as np


def sdf_randlankar(sd, riktningar, *, negativ_insida=True, periodisk=False):
    """Return (inside mask, links); each link is (flat inside indices, theta, flat back indices,
    back_is_inside). Outside-of-array links are excluded; periodic=True wraps all axes.

    With negativ_insida=True, sd<0 is inside. Otherwise sd>0 is inside and sd<=0 is the wall.
    A zero sample belongs to the boundary. Input arrays and the returned fractions are not mutated.
    """
    sd = np.asarray(sd, dtype=np.float64)
    directions = np.asarray(riktningar, dtype=np.float64)
    if sd.ndim < 1 or sd.size == 0 or not np.all(np.isfinite(sd)):
        raise ValueError("SDF must be a nonempty finite array")
    if (directions.ndim != 2 or directions.shape[1] != sd.ndim or
            not np.all(np.isfinite(directions)) or np.any(np.abs(directions) > 1) or
            not np.all(directions == np.rint(directions))):
        raise ValueError("directions must have SDF.ndim components, each -1, 0 or 1")
    inside = sd < 0 if negativ_insida else sd > 0
    ids = np.arange(sd.size, dtype=np.int64).reshape(sd.shape)
    axes = tuple(range(sd.ndim))
    links = []
    for direction in directions.astype(int):
        if not direction.any():
            links.append((np.empty(0, np.int64), np.empty(0), np.empty(0, np.int64),
                          np.empty(0, bool)))
            continue
        neighbor = np.roll(inside, tuple(-direction), axes)
        neighbor_sd = np.roll(sd, tuple(-direction), axes)
        mask = inside & ~neighbor
        if not periodisk:
            for axis, step in enumerate(direction):
                edge = [slice(None)] * sd.ndim
                if step > 0:
                    edge[axis] = slice(-step, None)
                elif step < 0:
                    edge[axis] = slice(0, -step)
                else:
                    continue
                mask[tuple(edge)] = False
        idx = np.flatnonzero(mask)
        a, b = np.abs(sd.ravel()[idx]), np.abs(neighbor_sd.ravel()[idx])
        scale = np.maximum(a, b)  # avoid overflow in |sd_f|+|sd_s|
        theta = (a / scale) / (a / scale + b / scale)
        back = np.roll(ids, tuple(direction), axes).ravel()[idx]
        back_inside = inside.ravel()[back].copy()
        if not periodisk and len(idx):
            coords = np.array(np.unravel_index(idx, sd.shape)).T - direction
            back_inside &= np.all((coords >= 0) & (coords < np.array(sd.shape)), axis=1)
        links.append((idx, theta, back, back_inside))
    return inside, links
