# Experimental 3D scalar intervals with a separate geometry term

This extends the reviewed 2D module with constructively embedded tetrahedral
meshes, P1/P2 potentials, shared-face RT0 flux, exact source repair and exact
degree-two integration. SciPy proposes fields; its tolerance never replaces
the exact conservation/trace checks. Scalar inputs follow the 2D exact-input
contract. Modified copies of admitted meshes or geometry proofs are refused.

```python
from fractions import Fraction
from field_engine.experimental import guaranteed_scalar3d as c
from field_engine.experimental import guaranteed_geometry3d as g

mesh = c.structured_mesh(2, lengths=(2, 1, 1))
electrodes = c.box_electrodes(mesh)
v, q, timing = c.solve_fields(mesh, source=0, electrodes=electrodes,
                             potential_degree=2)
cert = c.certificate(mesh, v, q, source=0, electrodes=electrodes,
                     potential_degree=2, with_energy_moments=True)
family = g.verified_shear_family(mesh, Fraction(1,4096))
robust = g.field_shear_family_response(cert, family)
assert robust['lower'] >= 1.999 and robust['upper'] <= 2.001
```

The result bounds continuum resistance for **every** body x->x+beta*y*e_x,
|beta|<=1/4096, with the same finite electrode faces transported by that
injective map. A symmetric voltage field and Piola current give the positive
quadratic energy polynomials; exact volume moments close the parameter range.
The classical analytical box shortcut matches the interval and is cheaper.
This is a scoped API extension, without a new error-estimation theory claim.

For source Poisson with zero Dirichlet boundary use source=1 and omit
electrodes. Its observable is C=int f*u, not an untyped routing resistance.
Constant-coefficient convex bodies can separately use verified ellipsoid
support-plane enclosures, or `verify_grid_sdf` followed by
`uncertain_convex_compliance`. The latter verifies all grid samples against
exact triangles, freezes a hash-bound SDF model snapshot, and pays the proved
trilinear interpolation remainder. Pass the axes and samples you actually solve
on (`uncertain_convex_compliance(..., axes, sampled_sdf)`) so they are checked
against that hash. Caller retains `physical_acquisition_error`.
`lift_grid_potential` can lift a successful `los_sdf_poisson` field to the
admitted 3D domain; native lattice matrix energy is not itself this certificate.

Constructive mesh factories cover boxes, reentrant L-prisms, verified convex
triangle-surface fans, exact conforming edge bisection and positive affine maps.
Arbitrary tetrahedral input, hanging nodes, nonlinear material laws, transient
physics and peak/interface stress are outside this contract.

Native mesh-to-SDF probe depth, median wall error or sampled maximum is not a
global physical geometry guarantee. Distance uncertainty alone cannot bound
general finite-electrode resistance: an arbitrarily thin wall near an electrode
can disconnect it while the stored SDF nodes stay identical. A validated
topology/electrode/domain-map contract is required. Physical scan acquisition
and conductivity calibration remain UNKNOWN unless supplied separately.

No default engine solver or route contract is replaced. All intervals refer
to the explicitly modeled quantity, geometry, coefficient and boundary traces.
The implementation and 3D mathematical obligations await independent review.
