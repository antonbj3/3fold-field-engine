# Optional panel CCD certificate under deformation

`native/ccd_deform_certificate.hpp` is a standalone C++17/Eigen3 CPU prototype.
It extends the immutable grid/topology validation of the scene certificate and
uses the box/family support license of `ccd_box`. No solver is automatically changed.
The API is in namespace `deform_ccd`: `Panel(rest, faces, edges, sphere_radius)`,
`Family{x0, endpoints, box_radius, complete}`, `Panel::support`, and `Transport::run`.

A Panel owns its immutable topology and reference gradient inverses. It cannot be
reassigned. A Transport borrows the Panel, which must outlive it; construction from
a temporary Panel is rejected. The unchecked internal support path is private and
accessible only to the cache after it has transported the injection proof. The projected
reference grid must be a complete rectangular triangulation with the diagonal
specified by the native scene certificate. Its domain is convex. Current positions
may deform in all three dimensions. The symmetric projected Jacobian must be
positive definite throughout each face and every supplied endpoint box/branch.
This sufficient condition proves projected injectivity and therefore absence of
self-intersection of the zero-thickness panel. Overhangs can cause a decline.

The obstacle must be a **fixed ball centred at the origin**, or geometry already
proved to be enclosed by the specified ball and independently proved free of
self-intersection. This header accepts only a radius; it does **not** verify an
arbitrary obstacle mesh. The harness validates its fixed convex ball once. Obstacle
pose/topology changes require reconstructing the producer's obstacle proof/cache.

`Family::complete` defaults false. A producer must explicitly declare the finite
supplied trajectory family complete for its chosen model contract. This flag does
not prove coverage of a continuous dynamical solution. Every endpoint coordinate
can vary independently within +/-box metres. Each branch follows affine time
interpolation between its two endpoint boxes. Empty/incomplete/nonfinite families
return prefix zero and licensed=false. Coordinates beyond +/-1e6 m decline.

Output `prefix=a` proves freedom on [0,a) when a<1. `a=1` additionally proves the
closed endpoint. The result is a conservative prefix, **not an existence witness
or two-sided first-contact interval**. A decline requires a sound external CCD
fallback if the caller wishes to advance; it must never mean free. Changing
branches or uncertainty is included in every transport displacement check.

A proposed separating direction need not be accurate: only the outward verified
projection at all time/simplex/box corners accepts a step. A plane separating
all triangle vertices from the enclosing ball separates every material point.
The corner license belongs to this multiaffine projection, not to minimum distance.

Transport retains an anchor geometry, a lower injection eigenvalue margin and
lower separation margin. Fixed reference derivatives bound the gradient change
by K times the maximum projected vertex displacement. Hausdorff displacement is
bounded by the maximum full vertex displacement including coordinate boxes.
Distance to a fixed set is 1-Lipschitz. Expired margins trigger refresh; failure
to transport injectivity declines. This is conventional conservative clearance
transport. An equally informed control can and does perform the identical operation.

All elementary proof operations use nextafter outward bounds; build with
`-ffp-contract=off` and without unsafe/fast floating-point optimization. Fast-math
builds are unsupported. Approximate closest points only propose directions.
Regression cases cover folds, stale/invalidated caches, unsafe branches, enlarged
boxes, endpoint contact, incomplete families, bad topology and oversized numbers.
Mathematical/arithmetic arguments are hand-derived and pending independent review.

The lane benchmark simulates spring cloth with a radial contact penalty, mesh
spacing 25–50 mm, 0.5 ms timesteps. Spring/contact parameters are synthetic closures;
yarn microstructure, physical thickness, friction and continuous-time integration
error are not resolved. Measured primitive CCD gains do not establish a new method
or a general 10x gain versus equally informed conservative advancement.

## Explicit reserve routing for a full-step consumer

`Transport::run_or_fallback(family, reserve_callback)` accepts the cached
answer directly only when it proves the entire closed step (`licensed` and
`prefix == 1`). All other valid-family outcomes, including exhausted support
and an injectivity decline, call the supplied sound original CCD reserve once.
The callback receives the complete validated family and must cover all its
branches, boxes, obstacle pairs and self pairs; this header does not implement
the external solver. A nonfinite, out-of-range or unlicensed reserve result
returns a decline. Callback exceptions propagate and prevent advancement.
Incomplete/invalid families remain declined without invoking the callback:
checking only supplied branches cannot recover an omitted trajectory.
`run` is still a conditional prefix API for callers that can shorten a step.
