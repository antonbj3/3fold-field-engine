# Explicit model decisions and physical abstention

`certified_decision` composes exact mathematical model bounds into JA, NEJ or
OSÄKER, recording the margin, input binding and missing information.
The default `scope='PHYSICAL'` returns OSÄKER because there is no verified scanner
or physical model port. `scope='MODEL'` answers only the declared geometry family.
`physical_admission` is always false. A serialized Decision is a report; it is
never accepted as evidence by any function in this module. A Decision refuses
inconsistent fields (also via `dataclasses.replace`), and `is_issued(decision)`
is true only for the object this module returned. A certain MODEL status still
lists the physical premises it lacks in `missing`.

```python
from fractions import Fraction as Q
from field_engine.experimental.certified_decision import (
    box_resistance, surface_plane_contact, scan_resistance,
)

cad = box_resistance(
    (2, 1, 1), conductivity=1, face_radius=Q(1, 10000),
    limits=(Q(199, 100), Q(201, 100)), scope='MODEL',
    units='declared model resistance',
)
assert cad.status == 'JA'

surface = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
overlap = surface_plane_contact(
    surface, [(0, 1, 2)], axis=2, plane_offset=Q(1, 100),
    coordinate_radius=Q(1, 1000), scope='MODEL', units='model length',
)
assert overlap.status == 'JA'  # a robust observed witness, all completions
assert overlap.lower is None  # unseen surfaces are unrestricted

physical = scan_resistance(surface, [(0, 1, 2)], units='source unit unknown')
assert physical.status == 'OSÄKER'
```

The box family has six independent planar face translations, full end electrodes
and a fixed positive scalar conductivity. It is constructed from parameters;
`face_radius` defines the family, and is not an inferred mesh or SDF error.
P1 potential, exactly balanced RT0 flux and rational energy integration provide
the numerical interval. Exact box transport adds the declared geometry interval.

Surface/plane overlap means `min signed plane gap < threshold`, a geometric
precondition, without mechanical force/contact closure. All represented triangles
are treated exactly. Unseen geometry can only reduce their minimum. Therefore a
penetrating observed witness can answer JA even with unknown solid topology.
NEJ needs `missing_axis_lower`, a bound on every unseen point AFTER uncertainty.
Every supplied vertex, including one that no face uses, enters the NEJ bound.
No finite supplied radius/support is a measured scanner bound in this API.
Touching a decision boundary returns OSÄKER. Exact endpoints and `exact_margin`
retain the proof when a positive margin underflows binary64.

Dependency: a007c990df606be11b51218d7781ec937593396e plus the independently reviewed
GARANTI3D patch (byte-identical to commit 6077d79 on falt-garanti3d-20261001). Method novelty and speed superiority are not established.
