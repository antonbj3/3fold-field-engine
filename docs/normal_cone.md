# Finite normal-cone certificates

`cone_chart` constructs `-N d <= relaxation`, `|d_j| <= 1`, and one signed
coordinate at least 1. For any nonzero d, divide by its largest absolute
coordinate. At least one of the 2n charts then contains it. A verified exact
infeasibility witness for every chart excludes every nonzero direction.
`verify_no_direction` rebuilds every row from the supplied normals and checks
sparse row indices, weights, chart identity and complete coverage. It reuses
the existing `_farkas_check`; numerical solver status is never a certificate.

`verify_direction` rejects zero and checks all normal rows exactly. An
antipodal pair can admit a boundary direction, so a zero-sum positive normal
combination alone does not exclude all nonzero directions. Normalization,
rounding-envelope derivation, mesh binding, measurement and swept assembly
clearance remain separate. An envelope bounds each row action at max-norm one.
Float proposals must be encoded explicitly as rationals before checking.

The Lean files reuse `LinearDual` and provide six synthetic opposite-axis
chart certificates plus a complete analytic opposite-axis exclusion theorem.
There is no application geometry in those fixtures. Compile LinearDual to an
olean on LEAN_PATH, then compile ChartExamples with a compatible Mathlib/Lean.
The general max-norm cover, parser and normal-source binding are checked by
Python and mathematical reasoning; they are not a general Lean theorem here.

Placement: field experimental Farkas API. A normal cone describes local
halfspace admissibility. A full assembly insertion path is another consumer.
