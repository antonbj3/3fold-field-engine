"""Certified binary32 filters for the existing vertical-ray and SDF contracts."""
import math
import numpy as np

from .adapters import RayTriangle as _OldRayTriangle, SDF as _OldSDF
from .fpband.raytri import raytri, HIT, MISS

U, ETA, U64, ETA64 = 2.0**-24, 2.0**-150, 2.0**-53, 2.0**-1074
INFL = 1.0 + 2.0**-40


def input_band(x):
    v = float(np.float32(x))
    return v, (U * abs(v) + ETA) * INFL


def add(a, b):
    v = float(np.float32(np.float32(a[0]) + np.float32(b[0])))
    return v, (U * abs(v) + a[1] + b[1]) * INFL


def sub(a, b):
    return add(a, (-b[0], b[1]))


def mul(a, b):
    v = float(np.float32(np.float32(a[0]) * np.float32(b[0])))
    e = U * abs(v) + ETA + abs(a[0]) * b[1] + abs(b[0]) * a[1] + a[1] * b[1]
    return v, e * INFL


def _max_band(pairs):
    pairs = tuple(pairs)
    if any(
        not math.isfinite(value)
        or not math.isfinite(band)
        or band < 0.0
        for value, band in pairs
    ):
        return 0.0, 1.0
    return (
        max(value for value, _ in pairs),
        max(band for _, band in pairs),
    )


def local(point, frame):
    """Bound rows @ (point-origin), including f64 arithmetic and f32 reduction."""
    deltas = []
    for x, c in zip(point, frame.origin):
        w = float(x) - float(c)
        deltas.append((w, (U64 * abs(w) + ETA64) * INFL))
    values, errors = [], []
    for row in frame.rows:
        terms = []
        for r, (w, ew) in zip(row, deltas):
            p = float(r) * w
            ep = (U64 * abs(p) + ETA64 + abs(float(r)) * ew) * INFL
            terms.append((p, ep))
        total, etotal = terms[0]
        for p, ep in terms[1:]:
            total += p
            etotal = (U64 * abs(total) + ETA64 + etotal + ep) * INFL
        v, e = input_band(total)
        values.append(v)
        errors.append((e + etotal) * INFL)
    return tuple(values), tuple(errors)


class SDF(_OldSDF):
    def cheap(self, query):
        node, point, frame = query
        pv, pe = local(point, frame)
        if node["op"] == "halfspace":
            terms = [mul((pv[i], pe[i]), input_band(node["normal"][i])) for i in range(3)]
            return sub(add(add(terms[0], terms[1]), terms[2]), input_band(node["offset"]))
        if node["op"] == "box":
            vals = []
            for i in range(3):
                d = sub((pv[i], pe[i]), input_band(node["center"][i]))
                vals.append(sub((abs(d[0]), d[1]), input_band(node["half_extents"][i])))
            return _max_band(vals)
        raise ValueError(node["op"])


class RayTriangle(_OldRayTriangle):
    def __init__(self, fma=False):
        self.fma = fma

    def banded(self, query):
        """Return the raw HIT/MISS/UNC band record with t and Et."""
        triangle, origin, _, frame = query
        def row(x):
            return np.asarray(x, dtype=np.float64).reshape(1, 3)
        return raytri(*(row(v) for v in triangle), row(origin),
                      row((0., 0., 1.)), row(frame.origin), self.fma)

    def cheap(self, query):
        _, _, tmax, _ = query
        if not math.isfinite(float(tmax)) or float(tmax) < 0:
            return (0.0,), (1.0,)
        result = self.banded(query)
        decision = int(result["dec"][0])
        t, et = float(result["t"][0]), float(result["Et"][0])
        if decision == MISS:
            return (-1.0,), (0.0,)
        if math.isfinite(t) and math.isfinite(et):
            lower = math.nextafter(t - et, -math.inf)
            upper = math.nextafter(t + et, math.inf)
            if lower > float(tmax):
                return (-1.0,), (0.0,)
            if decision == HIT and upper <= float(tmax):
                return (1.0,), (0.0,)
        return (0.0,), (1.0,)
