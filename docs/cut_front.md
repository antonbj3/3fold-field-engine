# Constructive scalar cut-front operator

`field_engine.experimental.cut_front` constructs a zero-thickness crack on a
fixed background grid, preserving area and separating its two field traces.
This is a generic scalar mode III elasticity/conductivity operator. It contains
no tissue calibration, needle-force closure, friction anchors or edge-radius API.
Lengths are in m, shear moduli in Pa, energy per thickness in J/m and finite-step
energy release in J/m². See `cut_front_theorems.md` for the precise guarantees.
The existing `guaranteed_scalar` arithmetic is reused without changes.

## Cut front

`evaluate(nx,a,mu=(1,1,1),aggregate=True,independent=False,certify=True,yc=0)`
returns `(mesh,v,psi,moments,certificate,cost)`; keyword-only options follow mu.
The constructive family is the rectangle (0,4)×(-0.5,0.5) m with a horizontal
zero-thickness mode III edge slit of length 0<=a<4 m. The slit lies on a
background row; two traces separate behind its tip. Total area stays exactly
4 m². `nx` counts cells along length; `ny=nx/4`. Mesh construction accepts
multiples of four >=8; the three-layer solver/certificate requires interfaces
at y=±1/8 to align, hence `nx` a multiple of 32. `yc` must be an interior row.
Full separation, off-row cuts and unresolved binary64 tip locations are refused.
With default eta=1/1000 and h=4/nx, a tip less than eta*h from the right outer edge
would merge stream nodes prescribed as 0 and 1, so `solve_fields` refuses it
as conflicting Dirichlet data. `aggregate=False` can propose fields for a
resolved terminal ligament; no theta-uniform conditioning bound covers it.

`solve_fields` proposes displacement and stream fields. `exact_moments` checks
exact boundary traces and unit-current identity, integrating rational triangles.
Only generated immutable mesh/moment objects are admitted. `energy_bounds`
uses E=integral(mu*|grad(v)|²) and D=integral(|q|²/mu), q=(-psi_y,psi_x):
`Pi in [Delta²/(2D),Delta²*E/2]`. Parameters mu are three positive shear moduli
in Pa and Delta is positive displacement in m (keyword `delta`). Thus the
certificate covers approximation, aggregation and algebraic solve errors for
this declared scalar PDE; physical model error remains UNKNOWN.

`release_bounds(left,right,da)` composes generated energy certificates for
the same material, displacement and outer geometry with exactly nested crack
lengths. It returns a finite-step mean energy release, not a local derivative.
The sum of endpoint energy widths is divided by da: historical small-step
widths were about 813% of stationary reference, missing the 10% budget.
The separate existing `guaranteed_energy_difference` API needs common-projector
Gram/metric hypotheses, which are not established for these changing crack
spaces; this module does not infer them or duplicate that algorithm.

This is **classical conforming P1 FE with local tip subdivision**, with the
same answers and guarantees as the equally informed FE control. In the prior
N64/N128 sweeps, 31 positions over three cells in 0.1h steps gave **309.2–380.9×
less G sawtooth than element deletion on the same grid**. That is neither a
gain over classical FE nor a runtime/full-cost speedup; deletion was cheaper
and lacked the certificate. Very short edges are aggregated before float
assembly, using exact grouped gradients to avoid cancellation.

The homogeneous stationary strip reference `Gss=mu*Delta²/(2H)` requires
stationary far fields. It is not exact for every finite slit. Three material
regions are synthetic continuum inputs. The module supplies no fracture
toughness or constitutive evolution law. Vector elasticity,
arbitrary SDFs, 3D, dynamics and irreversible history are outside its scope.
For the declared layers, `release_bounds` reports `stationary_Gss_exact` and
normalizes `relative_width_at_Gss` by `Delta²/(2*S)`, with
`S=sum(h_r/mu_r)` and `h=(3/8,1/4,3/8)` m. This layered reference follows
from constant far-field shear stress; it is a derived comparison scale,
not an exact finite-crack solution. Energy and release bounds do not use it.
Reference DOI
[10.1103/PhysRevLett.87.045501](https://doi.org/10.1103/PhysRevLett.87.045501).

```python
from fractions import Fraction as F
from field_engine.experimental.cut_front import evaluate, release_bounds

left = evaluate(32, F(2))
right = evaluate(32, F(17, 8))
mean_release = release_bounds(left[-2], right[-2], F(1, 8))
```
