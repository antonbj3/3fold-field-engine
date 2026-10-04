"""ACCEPTED review U380: mass properties of edited meshes and declared CSG leaves.

Mesh coordinates and outputs use mm, angles use radians. CSG uses m and
radians. Mesh derivatives are analytic for the discrete polyhedron, without
roundoff enclosure. CSG derivatives have empirical or uncalibrated error only.
"""
from .shape_massprop import shape_massprop
from .femur_edit import osteotomy

__all__ = ["shape_massprop", "osteotomy"]
