# Guaranteed scalar error on a represented polygon

Experimental; independent scientific review pending. Requires a conforming2D
triangle disk, matching positive scalar coefficient and constant body source.
All vertices are finite floats interpreted as exact dyadics, or explicit
Fractions. Holes,3D elasticity, resonant covariance and peak stress are outside
the contract. No continuum geometry guarantee is inferred from sampled SDF.

```python
from field_engine.experimental.guaranteed_scalar import (
    Mesh, solve_poisson_fields, poisson_certificate, scalar_readout,
    lift_grid_potential)
mesh = Mesh.admit(xy, triangles)
v, psi = solve_poisson_fields(mesh, source=2.0)
certificate = poisson_certificate(mesh, v, psi, source=2.0)
torsion = scalar_readout(certificate)  # Prandtl J=C for source2
# torsion['absolute_error_upper'] covers the true represented-polygon J
# around torsion['estimate'], including errors in the proposed sparse solves.
```

CUTRAND integration: after successful `los_sdf_poisson(sd,h,rhs=2)`, provide the
exact represented polygon triangulation separately. Use
`v=lift_grid_potential(mesh,axes,field,system['free'])`, then the same certificate
with `psi` from the conforming constructor. The polygon's exact zero trace and
the source-balanced flux close the PDE guarantee. No claim equates that polygon
with an arbitrary sampled SDF/CAD surface. Physical/representation error stays
`UNKNOWN`. `scalar_error_upper(certificate,raw_grid_integral)` bounds the error
of the raw grid integral by its maximum distance to the delivered interval,
with outward arithmetic.

`resistance_certificate` accepts continuous potential/stream nodal values and
four oriented boundary arcs bottom/right/top/left; the required traces are
potential0/1 on left/right and stream0/1 on bottom/top. It returns sheet resistance
in reciprocal sheet-conductance units. It does not certify routing graph point
electrodes or the pitch-dependent EDT coefficient. The exact proof is the
Dirichlet/Thomson complementary-energy principle, with no asymptotic multiplier.

Mesh admission, Fraction integration and dual construction have a measurable
cost. This module does not promise cheaper full work than conventional adaptive
FEM. Native point stress needs a different continuous observable and dual.
