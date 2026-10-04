# Experimental continuum instability witnesses

`field_engine.experimental.continuum_stability.rod_certificate` checks a
polynomial direction in a fixed incompressible Neo-Hooke rectangular solid.
Geometry, shear modulus and compression parameter accept exact integers,
rational strings or Fraction values. A vector polynomial is encoded as three
maps from `"X_power,Y_power,Z_power"` to exact rational coefficients.

Both complete end faces have prescribed affine displacements, the sides are
traction-free, F=diag(t²,t⁻¹,t⁻¹), 0<t≤1, and pressure=mu/t². The checker
verifies zero end traces and tangent incompressibility exactly, integrates
the constrained second variation over the whole volume and returns:

* `PROVED_UNSTABLE` if a valid direction has a strictly negative variation;
* `OSAKER` otherwise. Positive finite-dimensional data never certifies stability.

The `full_space_gradient_floor` is an analytical lower bound over all 3D
tangent perturbations. It is nonpositive for these compressive states and
cannot certify positive coercivity. `continuum_gradient_infimum_upper` is
an upper bound supplied by the witness, not a lower eigenvalue bound.
`verify` recomputes the entire result and must be used on imported records.
Invalid supports, non-solenoidal fields and inexact floats are rejected.
Natural side boundary conditions do not impose zero trial-field traction.

This is a mathematical model check, not experimental material calibration,
a general mesh backend, a positive-stability certifier or an exact critical
load enclosure. Classical exact Rayleigh methods provide the same guarantee.
The scalar arch/sphere scopes stay as documented in hyperelastic_energy.md.
This module accepts three-dimensional polynomial perturbations and imposes
both prescribed end traces and tangent incompressibility exactly.
