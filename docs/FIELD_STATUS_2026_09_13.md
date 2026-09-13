# Field lane status, 2026-09-13

The staging checkout contains the completed field work and diagnostic relocation
through `a4feafd`. Nothing has been pushed. Numerical development stopped at the
18:45 local freeze; subsequent changes concern paths, evidence and documentation.
README and frozen numerical references are unchanged.

| Scope | Recorded result | Limitation / next prerequisite |
|---|---|---|
| Prepared owned field service | All five original gates on RTX 5070; relocation rerun warm L 1.295530--1.479104 ms, complete outputs exact. | L4 still fails the 2 ms latency gate. Context construction is separate from owned warm queries. |
| Non-watertight winding | Ordered implementation passes six synthetic gates, maximum discrepancy from libigl 2.67841304690819e-15 within 1e-12. | Original open/reversed runs retain full-value repeat failures despite identical classifications. No general production-input validation claim. |
| Free-form field to loft | Six gates; 17 lofts and 56 operations, one valid solid, volume error 0.0153489887501 below 0.412822907959. | Evidence covers the recorded generated fixture. |
| Adaptive blocks and canonical inputs | Bracket controls and dense-source release pass; canonical cross-host report and all 5531960 input/output bytes exact. | Original face-order-dependent cross-host report failure remains separate. |
| Field/recipe oracle | Three gates; four zero-difference fixtures and two detected mutations, repeated full captures. Relocated entrypoint reproduces 437760 bytes exactly. | Sampled regression oracle, not a universal equivalence proof. |
| Contiguous tile reader, queue 44 (`7fe7a94`) | Original seven gates pass locally and twice on Modal; all 10142560 array bytes exact; 24 sanitizer cases clean on both hosts. | Gain versus original native reader is only 1.004--1.025x over the full operation; final Python packing dominates. Original 1.241598x negative versus Python retained. |
| Two-hit compaction, queue 49 (`f28bde9`) | Original five L4 stage gates twice; 6.325--6.431 ms versus paired general sorter 9.341--9.675 ms, 1.463--1.505x. All 70 arrays / 7192056 bytes exact per process; 17 stage controls and 11 sanitizer cases pass. | Bounded two-hit specialization; original wider-row/dtype fallback and GPU kernels retained. |

## Verification and hygiene

[RUNNING.md](RUNNING.md) is the authoritative per-module status and command list.
It contains 284 unique live rows: 149 engine, 19 supporting and 116 diagnostic
rows. Status totals are 168 VERIFIED-FRESH, 56 OWN-GATE-FAIL and 60 SYNTHETIC-ONLY;
there are no CUDA-ONLY rows. Diagnostic Python/native sources now live under
`probes/field_engine/`, with source transitions recorded in
[probe_relocation.json](../reports/probe_relocation.json).

The existing full CPU pytest run passes 62 tests, skips five CUDA-dependent tests
and reports four numerical-fit warnings in 516.13 seconds. The existing 92
verifier controls pass. CSG, recipe and prepared CPU selections pass their
original aggregate gates. The final complete local geometry selection passes
all five aggregate gates across six rows, including the relocated loft chain;
full reports and captures match exactly. See the
[complete geometry receipt](../reports/probe_relocation_geometry_complete.json). The public prepared native build and three local GPU
probes pass all four aggregate gates with 85558530 full capture bytes exact.
See [relocation validation](../reports/probe_relocation_validation.json) and its
companion reports for environment, scope and complete results.

A legacy substrate self-test exits zero even when its expected float32 variation
gate is false on CPU. Original and relocated sources reproduce the same report
excluding timing. Its current row is OWN-GATE-FAIL; pytest success alone does
not establish its internal gate. The historical GPU artifact remains unchanged.

Complete repository verification is **not accepted**: only 37 of 168 fresh rows
are explicitly mapped, 131 are unmapped, and three GPU rows are deferred in CPU
mode. Native cross-host sketch and full float64 density failures remain recorded;
the explicit Haswell plus NumPy feature-restriction profile reproduces the
recorded recipe group, but is not a new global default. No tolerance was relaxed.

## Work retained for a later numerical window

Queues 36 and 38/42 already have persistent/fused implementations and matching
recorded lifecycle gates; they must not be counted again as stateless gains.
After queues 44/49, the measured CPU priorities are first-use LBM calibration,
FEM sparse solves, LBM equilibrium/streaming and topology descent. Forty isolated
imports measure diagnostic process startup; removing isolation would change the
purpose of that check. Measurements and bounds are in
[hunt3_cpu_profile.json](../reports/hunt3_cpu_profile.json). No additional
implementation, dispatch study, replay harness or verifier suite was started.

All field GPU jobs have ended. The local GPU lock was released at 16:50:01 UTC;
no device settings changed. Source/document scrub patterns have zero matches.
The remaining release decision must retain the numerical failures, limited
coverage and fixture scope above rather than infer readiness from a clean tree.
