# Continuous projected facet gap

`field_engine.experimental.projected_gap.triangle_gap(upper, lower, unit='mm')`
returns the exact minimum of the affine height difference on the closed xy
overlap. The intersection vertices are contained facet vertices or edge
intersections; an affine function attains its minimum there. Orientation signs
are exact. Parallel edges need no division; their intersection endpoints are
already included. Zero projected area returns UNKNOWN. Empty overlap returns
EMPTY. `surface_gap` exhaustively checks all pairs and propagates UNKNOWN.

Inputs are finite int, Fraction, decimal/rational string or binary64 float.
Floats denote their exact represented dyadic values. The `exact` result is a
Fraction. `lower`/`upper` are outward binary64 bounds, including underflow and
overflow. Projection error, coordinate uncertainty, mesh acquisition and 3D
continuous collision are outside this operation. Triangle tessellation is the
resolved scale. No force/stiffness law is imported.

Placement: alongside the experimental geometry contract and SDF operators.
For PORT v1, construct `Gap(r.exact, r.exact, Unit(r.unit), source,
assurance=Assurance.MODEL_BOUND)` after a CERTIFIED result. Preserve source hash
and the projection/facet condition in the witness. EMPTY emits no contact;
UNKNOWN emits a blocked unknown gap, never an open contact. Physical status
remains UNKNOWN. The consumer bridge is separately checked against PORT v1;
PORT v1 must first be integrated before using that adapter. Conversion mm to m
is exactly division by 1000. Gap is length; solver bias `Scene.b` is m/s and
requires an explicit timestep/contact convention.
