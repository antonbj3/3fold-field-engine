"""Emulated binary32 arithmetic (numpy) + constants for the a-posteriori bands.

All algorithm arithmetic is done on numpy float32 arrays (IEEE-754 binary32,
round-to-nearest-even, gradual underflow).  A correctly rounded binary32 FMA
is emulated with an exact float64 product, TwoSum and a midpoint fix-up.
All *bound* arithmetic is done in float64 on non-negative terms and is
inflated by INFL (see RESULTS.md, lemma L0).
"""
import numpy as np

F32, F64 = np.float32, np.float64
U = 2.0 ** -24          # unit roundoff binary32
ETA = 2.0 ** -150       # half the smallest binary32 subnormal
U64 = 2.0 ** -53        # unit roundoff binary64
INFL = 1.0 + 2.0 ** -40  # covers <= ~2^12 float64 roundings per bound stage


def a64(x):
    """|x| as float64 (exact for float32 inputs)."""
    return np.abs(np.asarray(x).astype(F64))


def fma32(a, b, c):
    """Correctly rounded binary32 fma(a, b, c) for float32 arrays (no overflow)."""
    a64_, b64_, c64_ = a.astype(F64), b.astype(F64), c.astype(F64)
    p = a64_ * b64_                       # exact: 24+24 bits <= 53, exponent >= -298
    hi = p + c64_
    bb = hi - p
    lo = (p - (hi - bb)) + (c64_ - bb)    # TwoSum: p + c == hi + lo exactly
    r = hi.astype(F32)
    r64 = r.astype(F64)
    diff = hi - r64                        # exact
    alt = np.nextafter(r, np.where(diff > 0, F32(np.inf), F32(-np.inf)).astype(F32))
    # An overflowed r is ±inf; 2*diff == alt - r64 can then hold spuriously (both -inf), so never fix up.
    mid = np.isfinite(r) & (diff != 0) & (2.0 * diff == (alt.astype(F64) - r64))
    fix = mid & (lo != 0) & (np.sign(lo) == np.sign(diff))
    return np.where(fix, alt, r).astype(F32)


def validate_f32_ops(n, seed):
    """Compare numpy binary32 ops and fma32 against exact rounding (gmpy2/MPFR)."""
    import gmpy2
    from gmpy2 import mpq, mpfr
    rng = np.random.default_rng(seed)

    def rnd_f32(k):
        # mix of normal, subnormal, tiny-exponent and crafted values
        m = rng.integers(1, 2 ** 24, size=k).astype(F64)
        e = rng.integers(-172, 40, size=k)
        s = rng.choice([-1.0, 1.0], size=k)
        x = (s * m * np.exp2(e.astype(F64))).astype(F32)
        return x
    a, b, c = rnd_f32(n), rnd_f32(n), rnd_f32(n)
    # craft fma double-rounding hazards: c = -round(a*b) + small
    k = n // 4
    c[:k] = (-(a[:k].astype(F64) * b[:k].astype(F64))).astype(F32)
    # double-rounding hazards: a*b ~ half-ulp(c) with a tail below float64 precision
    ce = np.frexp(np.abs(c[k:2 * k]).astype(F64))[1] - 1
    sgn = rng.choice([-1.0, 1.0], size=k)
    a[k:2 * k] = (sgn * np.exp2((ce - 24).astype(F64)) * (1 + 2.0 ** -23)).astype(F32)
    b[k:2 * k] = (1 - rng.choice([1.0, -1.0], size=k) * 2.0 ** -23).astype(F32)
    ctx = gmpy2.ieee(32)
    ops = {"add": a + b, "mul": a * b, "div": a / np.where(b == 0, F32(1), b),
           "sqrt": np.sqrt(np.abs(a)), "fma": fma32(a, b, c)}
    ops["naive_fma_double_rounding"] = (a.astype(F64) * b.astype(F64) + c.astype(F64)).astype(F32)
    bad = {kk: 0 for kk in ops}
    checked = 0
    with gmpy2.context(ctx):
        for i in range(n):
            A, B, C = mpq(float(a[i])), mpq(float(b[i])), mpq(float(c[i]))
            Bd = B if b[i] != 0 else mpq(1)
            ex = {"add": A + B, "mul": A * B, "div": A / Bd, "fma": A * B + C,
                  "naive_fma_double_rounding": A * B + C}
            for kk, v in ex.items():
                r = float(mpfr(v))
                if abs(r) >= 3.4028235677973366e38:
                    continue
                if r != float(ops[kk][i]):
                    bad[kk] += 1
            r = float(gmpy2.sqrt(mpfr(abs(A), precision=200)))  # sqrt rounded in ieee32 ctx
            if r != float(ops["sqrt"][i]):
                bad["sqrt"] += 1
            checked += 1
    return {"n": checked, "mismatches": bad}
