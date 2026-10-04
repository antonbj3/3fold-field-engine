# Optional native full-scene CCD certificate

`experimental/native/ccd_scene_certificate.hpp` supplies `scene_cert::certify(X0,X1,E,F)`.
The result is either a sound full closed-step `safe=true` or a conservative refusal.
It is a header-only C++17/Eigen3 prototype, tested against the fixed CCDBAND base.
There is no automatic compilation, Python public API, solver installation or dispatch change.

The accepted family is an explicitly verified planar rectangular tiling per connected
component, exact common translation per component, and certified separation of all
component pairs by a finite dyadic time cover. All coordinates must convert exactly
to a bounded lattice (scale 2^40, |integer| <=2^50). Mesh edges must exactly equal
the face-induced edges. Primitive pairs with a common topological vertex are excluded.
The model is linear trajectories of zero-thickness surfaces; it does not model cloth
thickness, fibers, friction, physical uncertainty or nonlinear time interpolation.

On refusal, a caller must retain its original complete CCD control on ORIGINAL input.
Refusal has no safe-prefix guarantee. Scalable CCD's conservative prefix is distinct
from CCDBAND's exact contact existence and two-sided TOI interval contract. Retain
that separate reserve if the consumer requires exact contact time.

The benchmark includes initial topology validation per query, native full-scene GPU
broadphase/transfer/narrowphase fallback and exact public-query correctness checks.
The strongest conventional control also removes globally shared velocity exactly
before Scalable CCD. Claimed speedups must be compared to this control and must state
whether context/initial setup is charged. A synthetic gain is not a general benchmark
gain or method novelty. Independent proof review is required before product adoption.

Regression invocation: `python -m unittest discover -s tests -p test_ccd_scene_certificate_native.py`.
Set `EIGEN3_INCLUDE_DIR` if needed. Requires g++/Eigen3; absence is an explicit skip.
