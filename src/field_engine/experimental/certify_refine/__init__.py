"""Experimental filters for represented binary inputs.

The vertical ray/triangle filter uses the Woop–Benthin–Wald sequence and
the a posteriori bands of CLOUD-F-BAND-PROOF (hand proof R0–R9 plus exact
empirical checks). FMA and non-FMA sequences are supported. Halfspace and
box SDF signs use operation-wise bands including frame reduction and
subnormal rounding. Thickness retains its band in the normal f32 range.

The guarantee assumes IEEE round-to-nearest with gradual underflow and
the specified operation sequences. No particular GPU, FTZ/DAZ setting,
compiler contraction, acquisition error or physical model is covered.
"""
from .certify_refine import Filter, Frame, Statistics, decide, INNE, UTE, OSAKER
from .adapters import Thickness
from .proven_adapters import SDF, RayTriangle

__all__ = ["Filter", "Frame", "Statistics", "decide", "INNE", "UTE", "OSAKER",
           "SDF", "RayTriangle", "Thickness"]
