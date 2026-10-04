"""Experimental U377 thickness between triangulated surfaces in mm.

Normal ray and nearest surface distances are geometric estimates. No physical
or numerical error enclosure is returned. RT hardware has not been verified.
"""
from .thickness import thickness_field, moller_batch

__all__ = ["thickness_field", "moller_batch"]
