# Conditional common-error bounds

`projector_difference_bounds(R0,R1,C,m,M)` bounds the difference Q0-Q1
of squared error energies when R0/R1/C are the exact residual Gram entries
in one reference metric and `m A0 <= A1 <= M A0`. Requires a common
homogeneous trial space and equilibrated residuals. The scalar function
checks Gram positivity and coefficient ordering; it does not admit a mesh,
verify an external tensor inequality, or establish physical applicability.

`energy_difference_interval(J0,J1,R0,R1,C,m,M,step)` composes those bounds
with E_i=J_i-Q_i/2 and returns binary64 endpoints rounded outward. `step`
must be positive. No nested-crack monotonicity or parameter-box coverage
is inferred. Identical residuals with identical metrics cancel exactly.

This is the algebraic kernel for a paired continuum certificate, not a
complete general crack API. The slit transport/admission remains a separate
research prototype. The underlying method is conventional hypercircle
polarization with spectral metric comparison.
