"""Exact negative primitive for fixed convex projected passage geometry.

Nonnegative balanced support directions eliminate every planar translation.
This module never promotes a fixed-projection result to a global orientation
decision. Rotations, continuous covers and physical geometry acquisition belong
to the caller's separately verified contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable


def _q(value: int | str | Fraction) -> Fraction:
    # isinstance, not type(): numpy.float64 subclasses float and would otherwise
    # pass as "exact" geometry with a zero error budget.
    if isinstance(value, float):
        raise TypeError("Use exact integers, decimal strings or Fraction values")
    return Fraction(value)


def _points(values: Iterable[Iterable[int | str | Fraction]]) -> tuple:
    out = tuple(tuple(_q(x) for x in row) for row in values)
    if not out or any(len(row) != 2 for row in out):
        raise ValueError("A nonempty set of planar points is required")
    return out


@dataclass(frozen=True)
class TranslationExclusion:
    status: str
    obstruction: Fraction
    robust_lower: Fraction
    scope: str = "fixed projected shapes, every planar translation"


def balanced_support_exclusion(
    inner: Iterable[Iterable[int | str | Fraction]],
    aperture: Iterable[Iterable[int | str | Fraction]],
    normals: Iterable[Iterable[int | str | Fraction]],
    weights: Iterable[int | str | Fraction],
    *,
    inner_error_upper: int | str | Fraction = 0,
    aperture_error_upper: int | str | Fraction = 0,
) -> TranslationExclusion:
    """Return a sufficient exact no-translation certificate, or UNKNOWN.

    Geometry is the convex hull of the supplied planar points. Error arguments
    bound Euclidean Hausdorff distance from each supplied convex hull. Length
    units must agree. The L1 normal norm bounds its Euclidean norm, so the error
    correction remains rational and conservative. UNKNOWN does not prove a fit.
    """
    a, b, ns = _points(inner), _points(aperture), _points(normals)
    ws = tuple(_q(w) for w in weights)
    if len(ns) != len(ws) or any(w < 0 for w in ws) or sum(ws) != 1:
        raise ValueError("Weights must be nonnegative and sum exactly to one")
    if any(sum(w * n[j] for w, n in zip(ws, ns)) != 0 for j in range(2)):
        raise ValueError("Support directions must balance exactly")
    errors = (_q(inner_error_upper), _q(aperture_error_upper))
    if any(e < 0 for e in errors):
        raise ValueError("Geometry error bounds must be nonnegative")

    def support(points, normal):
        return max(sum(v[j] * normal[j] for j in range(2)) for v in points)

    obstruction = sum(w * (support(a, n) - support(b, n)) for w, n in zip(ws, ns))
    normal_bound = sum(w * sum(abs(x) for x in n) for w, n in zip(ws, ns))
    robust_lower = obstruction - sum(errors) * normal_bound
    return TranslationExclusion(
        "CERTIFIED_NO_TRANSLATION" if robust_lower > 0 else "UNKNOWN",
        obstruction, robust_lower,
    )
