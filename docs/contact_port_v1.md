# Contact PORT v1

`field_engine.contact_port_v1` is the dependency-free canonical implementation.
The opt-in motion adapter imports these same classes. Install the field patch
and the matching field package before importing the motion adapter. Existing
solver entrypoints/defaults have no new dependency.

`Gap(lower, upper, unit, source, sigma, assurance, condition, blocked)` represents
signed surface clearance, positive apart. Units are `m` or `mm`, converted with
exact rational arithmetic. Binary floats are transported as their represented
rationals. This does not repair upstream rounding errors. `sigma` is the
projected joint gap sigma, not a deterministic radius; no missing sigma is set
to zero. Conditional bands retain their declared event and threshold.
`classify(threshold)` judges that threshold in the Gap's current unit; UNKNOWN
is returned on touching, unknown completion or conflicting completion sign.

`ContactCandidate` binds an ID, two participants, frame and Gap. Its optional
normal must be exactly unit length; an uncertified normal can remain absent.
`BranchSet` contains states, replayable witnesses and a completeness claim.
Status is UNKNOWN, EMPTY, UNIQUE or MULTIPLE, derived from completeness and
cardinality. OPEN/CLOSED apply to geometry. AMBIGUOUS maps to MULTIPLE;
OSAKER/OSÄKER/UNCERTAIN map to UNKNOWN. These meanings must not be collapsed.
All scalar equilibria, including unstable roots, are retained by default.

`Witness` stores kind, canonical JSON, canonical input SHA-256 and the original
source identity/hash/scope. `Port` roundtrips all branches, coverage, conditions,
latent affine gap laws and extra evidence (e.g. joint covariance). Both opaque
state and witness payloads are canonical JSON; the outer wire uses exact
rational strings. The structural JSON schema is `contact_port_v1.schema.json`.
The decoder rejects unknown fields/versions, missing provenance, unsupported
physical claims, escaping affine laws and noncanonical numeric fields.

A completeness flag in an imported file is a claim, not proof. Use
`replay_witnesses(branch_set, replayers)` before numerical admission. Replayers
must check the full model/input binding, all supports and global coverage;
record their exact code hash. Unknown witness kinds fail replay. The codec is
not a sandbox or a verifier for arbitrary untrusted code. Source hashes alone
establish identity, not independence or empirical accuracy.

`from_contact_band` requires external condition metadata and explicit frame/
participant binding because upstream ContactBand does not retain z, covariance
or normals. `from_univariate` replays the existing exact polynomial verifier.
`from_shell_equilibria` requires a full consumer verifier supplied by the caller;
checking roots alone is insufficient. `from_aligner_port` preserves all legacy
branches/source fields but reports UNKNOWN until coverage is replayed.

Physical status in v1 remains UNKNOWN. A scalar reduced shell certificate cannot
be promoted to a full aligner or an anatomical collision/force certificate.
