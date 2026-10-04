"""Unaudited U384 elastic cartilage contact; exploratory parameter grid only.

R, layer thickness and indentation are m; E and pressure are Pa; force is N.
The bonded isotropic linear elastic layer is contacted by a rigid sphere with
a frictionless upper surface. Grid discrepancy is not a uniform error bound.
Poroelastic time curves are model envelopes, not a validated biphasic solution.
"""
from .contact_api import cartilage_contact
from .regime import regime_report
from .poroelastic import force_time_envelope

__all__ = ["cartilage_contact", "regime_report", "force_time_envelope"]
