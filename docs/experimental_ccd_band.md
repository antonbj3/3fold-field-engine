# Experimental continuous collision decisions

`from field_engine.experimental.ccd_band import certify, certify_batch`

`certify(endpoints, "PT")` takes a `(2,4,3)` binary64 array: time endpoint,
vertex, coordinate; vertices are point,a,b,c. For `"EE"` they are a,b,c,d,
with edges a-b and c-d. `certify_batch` takes `(N,2,4,3)` and one primitive
kind. `certify` defaults to the projection filter plus the accelerated exact
reserve. In `certify_batch`, set `projection=True, accelerated=True` to enable
that path. `exact=False` enables only the float filter and returns unresolved
cases as OSÄKER. Status constants are SAFE="SÄKER", COLLISION="KOLLISION",
UNKNOWN="OSÄKER". A Decision also contains method and optional witness.

Only SÄKER permits a whole line-search segment. KOLLISION and OSÄKER restrict
it. The API does not compute forces, positive thickness, first impact time,
broad phase, or nonlinear trajectories. It includes initial contact, grazing,
and contact at t=1. Coordinates are the exact represented inputs in one
consistent unit. Keep inputs unchanged during a call.

NumPy is required. The exact cubic root reserve optionally uses SymPy; without
it those unresolved cases return OSÄKER. Fast float and dyadic integer answers
do not need SymPy. General coplanar motion may also remain OSÄKER.

The tested research implementation supplies a hand proof and regression tests;
independent scientific review and cloth integration are still pending.

# Soundness and explicit limits

This is a hand proof and executable regression evidence, pending independent review.
It is not a Lean theorem, a physics certificate, or a proof that a broad phase found
every candidate. Inputs are finite represented binary64 endpoints. Trajectories
are affine, triangles and segments are closed, thickness is zero, time is [0,1].

## Arithmetic license

For a correctly rounded binary32 operation with exact real result z and finite
rounded m, |m-z| <= u |z| + eta, u=2^-24, eta=2^-150 (gradual underflow).
Thus |m-z| <= (u |m| + eta)/(1-u) <= 2u |m| + 2eta.
For the normal range the eta term is unnecessary; for underflow the absolute
rounding error <= eta. Our implemented radius 2u|m|+eta covers both: if m is
normal, the relative bound suffices; if m is subnormal, eta suffices. All positive
radius arithmetic is rounded upward one binary64 ulp *after each operation*,
including products that underflow to zero. There are no long reductions.

The initial binary64 difference is enclosed with 2u64|d|+smallest_subnormal64;
the subsequent f32 cast error is bounded by an outward subtraction. Cast
overflow or any nonfinite band disables the float certificate. Input reduction
is included: the contract never silently replaces a world-space float64 by f32.

Addition adds both radii and the operation rounding error. Multiplication adds
|a_mid| r_b + r_a |b_mid| + r_a r_b and the operation rounding error. Negation is
exact. Every outward endpoint uses nextafter. Induction over the explicit
operation tree proves that every finite band encloses the exact expression.
This follows the existing represented-input and underflow discipline and a
filtered exact-reserve construction. It uses a more conservative 2u radius
and individual outward rounding rather than an unproved new global inflation.
No FMA, fast math, flush-to-zero, unchecked GPU port, or compiler contraction is
authorized by this proof.

## Geometric license for the float corners

PT F(t,u,v)=p(t)-a(t)-u(b(t)-a(t))-v(c(t)-a(t)), with u,v>=0,u+v<=1.
EE F(t,u,v)=a(t)+u(b(t)-a(t))-c(t)-v(d(t)-c(t)), u,v in [0,1].
Each coordinate is separately affine in t,u,v. Its value is a convex
combination of its eight cube corners. PT's square encloses the triangle's
barycentric domain, adding the fourth corner p-b-c+a. Hence if any coordinate
corner intervals have a common strict sign, F is never zero and contact is
impossible. We never apply the corner argument to squared distance or energy.

## Fixed-direction license

For any finite vector n, the projected separation of two convex primitives
is bounded by their extreme vertex projections. If every inter-primitive
n.(vertex_i-vertex_j) has a common strict sign at both time endpoints, its
affinity in time keeps that sign throughout. A fixed n need not be the exact
normal: all finite rounded directions are legitimate, possibly less useful.
The projection's dot products and endpoint differences retain their bands.

## Exact reserve license

At any PT or EE intersection, the determinant of three relative affine vectors
is zero. The determinant is a rational polynomial of degree <=3. If nonzero,
exact real-root isolation enumerates all roots in closed [0,1], including
multiple and endpoint roots. At a root, choose a nonzero projected 2x2
determinant. Coplanarity then makes projected Cramer coordinates the exact
3D barycentric coordinates. Numerator/denominator signs decide the closed
simplex/square inequalities without division. All-zero minors imply collinear
directions: endpoint-on-segment cross products and dot products decide
intersection, including zero-length segments and collapsed triangles.

Polynomial signs at irrational roots use exact rational interval evaluation.
A gcd sharing the isolated determinant root establishes an exact zero;
otherwise interval refinement determines the sign. Exhausting the explicit
budget returns UNKNOWN, never SAFE. If all enumerated roots are outside the
primitive domain, no intersection exists and SAFE is exact.

An identically zero determinant needs a different algorithm. The implementation
may prove contact at rational samples 0,1/2,1, and may decide static relative
geometry completely. Other coplanar motion returns UNKNOWN. Samples can
prove existence by exact membership; they can never prove absence.

## Scope down the resolution hierarchy

Uniformly rescaling or refining represented geometry preserves the arguments,
while f32 decisiveness may fall and the exact reserve may cost more. A gap below
the input format's resolution is not recovered. No acquisition-error interval,
cloth thickness, curved-surface deviation, nonlinear motion, force law, friction,
or atomic-scale physics is inferred. Scene safety requires all primitive pairs
and a line search that treats UNKNOWN as a restriction.

## Integer Bernstein license (v2)
Every finite binary64 coordinate is an integer divided by a power of two. Lift
all 24 coordinates to one common denominator and perform the same determinant
polynomial with integer addition/multiplication. No gcd, truncation, overflow
or input rounding occurs. For d(t)=d0+d1*t+d2*t^2+d3*t^3, three times its
cubic Bernstein coefficients are 3d0, 3d0+d1, 3d0+2d1+d2, 3(d0+d1+d2+d3).
Bernstein basis functions are nonnegative and sum to one on [0,1]. Thus strict
common sign excludes all determinant roots and proves SAFE. Exact contact at
0,1/2,1 may instead prove COLLISION. At 1/2 the integer coordinates x0+x1
represent a common doubled scale, preserving incidence; a zero determinant
and exact closed-primitive membership supply the witness. Failure of these
tests only invokes the existing reserve. This is standard polynomial
exclusion, not a claim of a new general CCD theorem.
