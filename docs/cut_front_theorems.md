# Cut-front model guarantees

All guarantees below concern the constructive scalar PDE and exact inputs.
Physical constitutive error is UNKNOWN. Floating-point solves propose fields;
feasibility and exact integrals establish the certificate.

## Geometry and partition

The rectangle is (0,4)×(-1/2,1/2), with slit [0,a]×{yc}, 0<=a<4.
The horizontal slit lies on an interior background row. A moving subcell tip
divides two parent triangles into positively oriented integration triangles.
Interior partitions are disjoint and cover the parents exactly. Coincident
crack-face coordinates have separate indices behind the tip, while the tip
and forward trace remain shared. No physical volume is removed: exact area is 4.
Topology and number of field traces change; this is a classical conforming
P1 space with local subdivision, rather than an unchanged finite-dimensional space.

## Complementary energy

For unit Dirichlet displacement v on bottom/top, let
E(v)=integral(mu*|grad v|²), and let psi be continuous P1 with values 0 on the
whole insulating boundary component (including both crack faces and tip),
and 1 on the right boundary component. The rotated gradient
q=(-psi_y,psi_x) is divergence free, has continuous normal flux across internal
edges and zero normal flux on the insulating boundary. Its total current is 1.
Exact integration verifies integral(grad v dot q)=1.

Dirichlet's principle and the complementary principle give 1/E<=R<=D,
where D=integral(|q|²/mu). Cauchy–Schwarz also proves E*D>=1.
For fixed displacement Delta>0, stored energy per out-of-plane thickness is
Pi=Delta²/(2R), hence Pi∈[Delta²/(2D),Delta²*E/2].
Reaction lies in [Delta/D,Delta*E]. These are continuum bounds for the declared
geometry, including approximation, aggregation and algebraic solve error.

Coordinates, supplied finite binary64 nodal values and material parameters are
interpreted as exact rationals. Gradients and triangle integrals are rational;
output endpoints round outwards using comparison to exact values. Exact 0/1
boundary traces are required. Generated mesh/moment identity guards reject
arbitrary replacements; callers must preserve the generated certificate metadata.

## Nested finite-step release

The same material, displacement, outer rectangle and yc with nested crack
lengths imply Pi_right<=Pi_left. For da=a_right-a_left>0:

max(0,(L_left-U_right)/da) <= mean_G <= (U_left-L_right)/da.

The code checks common scope and exact crack increment. This is mean release
over the finite interval, not a certified local derivative. The two endpoint
widths add and are divided by da. Thus narrow point estimates can coexist with
wide release intervals; reducing da can worsen the bound.
No common-projector residual premise is inferred across changing crack spaces.

## Positive operator and aggregation

Each triangle stiffness is a positive weighted Gram matrix, hence K>=0.
For a<4 the constructive mesh is connected: its only exact null mode is constant.
The one-hot aggregation map P uses every group and has full column rank,
so PᵀKP remains positive semidefinite with that single constant mode.
Dirichlet anchoring then makes the free operator positive definite.

When theta or 1-theta is smaller than eta=1/1000, short traces are aggregated.
Grouped gradients are added as rationals before conversion to binary64, avoiding
cancellation of large 1/theta terms. On the same reduced coordinates, comparison
with the endpoint triangulation gives
cL*Kref <= Kagg <= cU*Kref, with
cL=(1-eta)/(1+2eta)² and cU=1/(1-2eta)²+4eta.
The large triangles are affine endpoint images with ||A-I||<=2eta and the
aggregated small triangle contributes a positive term bounded by 4eta*Kref.
Material interfaces are aligned so each split parent keeps a single coefficient.
The bound is independent of the tiny theta fraction, but can depend on grid
spacing, coefficient contrast and eta; it is not a grid-independent promise.
The comparison requires an admitted endpoint triangulation with endpoint
length less than 4. When the right endpoint would be 4, aggregation identifies
the insulating tip (psi=0) with the right boundary trace (psi=1), and the
solver refuses those conflicting Dirichlet data. Resolved terminal ligaments
may instead use `aggregate=False`; this endpoint comparison supplies no
theta-uniform conditioning guarantee for them.
Unresolved binary64 tip locations are refused, never silently snapped.

## Regional parameters and stationary reference

For fixed feasible fields, P_r=integral_r|grad v|² and Q_r=integral_r|q|² give
E(mu)=sum(mu_r*P_r) and D(mu)=sum(Q_r/mu_r). Their nonnegative moments license
monotone endpoint bounds for energy in a positive material box. This does not
license corner extrema for the difference of two optimized energies.

An intact strip of layer heights h_r has compliance S=sum(h_r/mu_r), reaction
L*Delta/S and Pi=L*Delta²/(2S). This supplies an exact independent test.
The corresponding stationary energy release comparison is Delta²/(2S).
The reported relative interval width uses that scale, including material and
displacement scaling; `stationary_Gss_exact` records its rational value.
This layered extension is derived from constant shear stress in the intact
far field. It supplies no additional guarantee for a finite crack.
For a homogeneous stationary strip, Gss=mu*Delta²/(2H) needs stationary far
fields. It is not an exact finite-crack solution. Source DOI:
[10.1103/PhysRevLett.87.045501](https://doi.org/10.1103/PhysRevLett.87.045501).

Arbitrary curved/off-row cracks, full separation a=4, 3D, vector elasticity,
dynamics and irreversible constitutive history are outside the admitted family.
