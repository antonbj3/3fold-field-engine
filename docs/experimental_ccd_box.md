# Conditional continuous collision detection over uncertainty boxes

`Branch.from_arrays(centers, radii, skin_lower, skin_upper, id=...)` copies arrays
into immutable exact rational tuples. All coordinates and skin radii are metres.
Rational numerators/denominators are normalized to unbounded Python integers;
fixed-width NumPy integers cannot silently overflow certificate arithmetic.
Centers and radii have shape (2,4,3); skin endpoints have shape (2,4). PT order is
point, face vertices; EE order is the two endpoint pairs. Skin is a spherical
radius, not total fabric thickness. Full thickness must be divided by two by
the producer when that is the correct physical model.

`certify_boxes(branches, kind, complete=True)` resolves a **model** query.
Every endpoint coordinate may vary independently in its closed input box; each
world's endpoints are interpolated linearly on the normalized closed step [0,1].
Skin lower/upper endpoints are similarly interpolated and then interpolated in
barycentric coordinates. Inputs may instead enclose a correlated subset, but the
API certifies the larger independent product and can lose useful correlations.
For a material point on each primitive, contact is distance <= sum of skins.

SÄKER means all worlds in all branches are free throughout the step.
KOLLISION means every world of every branch has at least one contact, possibly
at a different time or different primitive point. OSÄKER means neither result
has been certified. A proved mixed-world reason establishes that both outcomes
are admissible; it is an information limit, rather than a subdivision timeout.
Mixed branch statuses also remain OSÄKER. Empty or incomplete input is OSÄKER.
An upstream solver must establish branch completeness and trajectory enclosure;
`complete=True` is a caller assertion, not validation of a nonlinear solver.

For OSÄKER, `safe_prefix` may certify the open interval [0,prefix); zero is empty.
SÄKER certifies the entire closed step. `prefix_float()` rounds down. A consumer
must not accept a prefix boundary without its own strict retreat/event policy.
The implementation reports a conservative prefix 0 for universal collision or
mixed-world witnesses; it does not claim an uncertainty-aware first-contact time.

## License at quantity level

For any finite exact direction n, simplex weights a,b and time t, the relative
position is F(t,a,b,delta). Endpoint coordinates appear affinely and all weights
are nonnegative. The support minimum is

    p(t,a,b) = sign*n·F_center - sum_i w_i sum_k |n_k| r_ik(t).

It is affine separately in t, a and b. A multiaffine scalar on an interval times
simplexes is a convex combination of its endpoint/simplex-vertex values.
Thus those values certify the entire domain. Upper skin h is also multiaffine.
Checking p>0 and p²>h²(n·n) at each of these vertices is equivalent there to
p-h*sqrt(n·n)>0, a multiaffine scalar. Positivity everywhere yields ||F||>h
by Cauchy-Schwarz. The **scalar projection minus skin**, rather than squared
distance, has the corner license. Directions are proposals, never inferred exact
normals; their accuracy affects only coverage.

As an alternative inclusion bound, each coordinate of F lies in the hull of its
licensed scalar bounds. Squared minimum distance to this enclosing coordinate
box is the sum of squared clamped distances per coordinate. If it exceeds maximum
skin squared, the cell is free. This uses set inclusion, not a distance corner
theorem. All arithmetic and strict comparisons use Python Fraction. No floating
point outward-radius guess is involved in the candidate certificates.

Time bisection and longest simplex-edge bisection cover their parent closed
domains. Separation of every leaf separates the union. Unprocessed/depth-limited
cells remain unresolved. Earliest unresolved time lower endpoint gives only an
open certified prefix. Smaller cells preserve validity without guaranteeing
termination near contact, tangency or mixed outcomes.

## Universal collision

1. At a fixed rational time and fixed admissible simplex points, the entire
relative box has maximum squared norm sum_k (|center_k|+radius_k)². If this is <=
minimum interpolated skin squared, every world contacts at that same witness.
Probe times/simplex points are **sufficient witnesses**; they do not establish
absence of contacts when no witness succeeds. Nominal projected barycentric
weights are exact proposals, each checked for membership before use.
2. For PT, exact interval arithmetic encloses endpoint signed tetrahedron volumes.
Opposite strict endpoint signs imply a zero by continuity for each world. The
moving triangle's projected orientation and all three projected point-edge
orientations are bounded strictly with the same sign over swept coordinate boxes.
This guarantees nondegeneracy and interior projected containment at every time.
At each world's volume zero, projection uniquely identifies a point in the face,
so the point contacts the face. The swept-box inclusion bound intentionally
handles the time-quadratic orientations; checking their two endpoints alone
would be invalid. Volume intervals are also inclusion bounds, not an unlicensed
corner claim on a time-cubic quantity.
3. Zero-box, zero-skin binary64 queries may use the reviewed `ccd_band.certify`.
Non-binary rational input never silently converts to that reserve; exact direct
witnesses remain legal. The inherited reserve's coplanar/refinement limitations
remain OSÄKER. This API does not replace that base.

## Provably mixed worlds

At fixed time/simplex coordinates, independent endpoint errors map onto the
entire relative coordinate box: each coordinate is a Minkowski sum of intervals
with nonnegative weights. Minimum norm from coordinate clamping <= maximum skin
proves existence of an admissible contact world. For a free-world witness, assign
all A endpoint errors along sign*n and B errors against sign*n, coordinatewise
at their bounds, and choose minimum skins. Licensed strict support over the whole
step proves this one consistent world free. Having both witnesses refutes both
universal outcomes, so OSÄKER is necessary on this input set. Probes are used as
existence proposals and do not imply that every mixed set will be detected.

## Scope and evidence

Geometry is a piecewise planar mesh. This is not a clinical/anatomical clearance,
physical cloth, contact-force or nonlinear trajectory certificate. PSF sigma
measures imaging resolution, not automatically a calibrated surface-error bound.
Neither a Gaussian law nor z*sigma guarantees deterministic physical containment.
Refinement must carry newly established coordinate/skin/branch enclosures.

The proof is a hand derivation with regression and external benchmark testing,
pending independent review. Fraction correctness and the Python runtime are
trusted. Standard interval arithmetic, simplex support and continuity are prior
operations; this module establishes an API capability, not general method novelty.
Unchanged Tight Inclusion plus this same reserve can match every final decision.
