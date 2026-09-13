"""Independent outward-float bounds for the frozen double barycentric tests."""
import numpy as np

SCALE_HEX = (
    '0x1.19799812dea11p-40',
    '0x1.5fd7fe1796495p-37',
    '0x1.b7cdfd9d7bdbbp-34',
    '0x1.12e0be826d695p-30',
    '0x1.5798ee2308c3ap-27',
    '0x1.ad7f29abcaf48p-24',
    '0x1.0c6f7a0b5ed8dp-20',
    '0x1.4f8b588e368f1p-17',
    '0x1.a36e2eb1c432dp-14',
    '0x1.0624dd2f1a9fcp-10',
    '0x1.47ae147ae147bp-7',
    '0x1.999999999999ap-4',
    '0x1.0000000000000p+0',
    '0x1.4000000000000p+3',
    '0x1.9000000000000p+6',
    '0x1.f400000000000p+9',
    '0x1.3880000000000p+13',
    '0x1.86a0000000000p+16',
    '0x1.e848000000000p+19',
    '0x1.312d000000000p+23',
    '0x1.7d78400000000p+26',
    '0x1.dcd6500000000p+29',
    '0x1.2a05f20000000p+33',
    '0x1.74876e8000000p+36',
)


def outward(values, upper):
    values = np.asarray(values, np.float64)
    rounded = values.astype(np.float32)
    needs = rounded.astype(np.float64) < values if upper else rounded.astype(np.float64) > values
    direction = np.float32(np.inf if upper else -np.inf)
    return np.where(needs, np.nextafter(rounded, direction), rounded)


def subtract(a, b, upper):
    result = outward(a.astype(np.float64) - b.astype(np.float64), upper)
    if not upper:
        opposite_zeros = (a == 0) & (b == 0) & (np.signbit(a) != np.signbit(b))
        # Directed subtraction returns -0 on cancellation; opposite zeros
        # retain the first operand's sign, as specified by CUDA Math 12.9.
        result = np.where((a == b) & ~opposite_zeros, np.float32(-0.), result)
    return result


def multiply(alo, ahi, blo, bhi):
    products = np.stack([x.astype(np.float64) * y.astype(np.float64)
                         for x, y in ((alo, blo), (alo, bhi), (ahi, blo), (ahi, bhi))])
    return outward(products.min(0), False), outward(products.max(0), True)


def bounds(triangles, queries):
    difference = triangles[:, :, :2] - queries[:, None, :]
    lo, hi = outward(difference, False), outward(difference, True)
    ulo, uhi = np.roll(lo, -1, axis=1), np.roll(hi, -1, axis=1)
    vlo, vhi = np.roll(lo, -2, axis=1), np.roll(hi, -2, axis=1)
    left_lo, left_hi = multiply(ulo[:, :, 0], uhi[:, :, 0], vlo[:, :, 1], vhi[:, :, 1])
    right_lo, right_hi = multiply(ulo[:, :, 1], uhi[:, :, 1], vlo[:, :, 0], vhi[:, :, 0])
    lower = subtract(left_lo, right_hi, False)
    upper = subtract(left_hi, right_lo, True)
    reject = (lower > 0).any(1) & (upper < 0).any(1)
    return lower, upper, reject


def fixtures():
    rng = np.random.default_rng(20260915)
    count = 50000
    scales = np.array([float.fromhex(h) for h in SCALE_HEX], np.float64)[rng.integers(0, 24, size=(count, 1, 1))]
    triangles = rng.uniform(-1, 1, (count, 3, 3)) * scales
    queries = rng.uniform(-2, 2, (count, 2)) * scales[:, 0]
    # Exact vertices, edge midpoints and adjacent representable coordinates.
    edge_triangles = triangles[:10000]
    edge = (edge_triangles[:, 0, :2] + edge_triangles[:, 1, :2]) * .5
    batches = [(triangles, queries), (edge_triangles, edge_triangles[:, 0, :2]),
               (edge_triangles, edge), (edge_triangles, np.nextafter(edge, np.inf)),
               (edge_triangles, np.nextafter(edge, -np.inf))]
    unit = rng.uniform(-1, 1, (10000, 3, 3))
    unit_queries = rng.uniform(-2, 2, (10000, 2))
    for translation in (1e6, 1e12 - 2):
        batches.append((unit + translation, unit_queries + translation))
    batches.append((unit * 1e-150, unit_queries * 1e-150))
    zero = np.zeros((16, 3, 3)); zero[::2] = -0.
    batches.append((zero, np.zeros((16, 2))))
    return np.concatenate([x for x, _ in batches]), np.concatenate([y for _, y in batches])


def frozen_edges(triangles, queries):
    diff = triangles[:, :, :2] - queries[:, None, :]
    u, v = np.roll(diff, -1, axis=1), np.roll(diff, -2, axis=1)
    return u[:, :, 0] * v[:, :, 1] - u[:, :, 1] * v[:, :, 0]
