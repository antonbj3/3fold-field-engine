# Optional exact univariate equilibrium certificate

`field_engine.univariate_equilibrium.certify(coefficients)` accepts ascending coefficients as exact integers, rational strings or Fractions. It returns `ENTYDIG`, `GRENMANGD` (including the empty set), or `OSAKER`. These refer to all REAL roots of the input polynomial, never automatically to a continuum or measured material. Singular roots and insufficient budgets remain `OSAKER`; a zero polynomial is an unresolved continuum.

```python
from fractions import Fraction as Q
from field_engine.univariate_equilibrium import certify, verify
c = (Q(1,1000), -Q(1,25), Q(1,1000), 1)
certificate = certify(c)
assert verify(certificate, coefficients=c)
assert certificate['complete'] and len(certificate['roots']) == 3
```

Always supply the query's expected `coefficients` to `verify`. A valid certificate for an old polynomial is not a certificate for a new material/load. The verifier checks exact rational arithmetic, strict interval-Newton inclusions, disjoint witnesses and a complete partition inside a global Cauchy bound. `verify=True` also accepts an honestly PARTIAL certificate; require `complete=True` separately before claiming a complete set. No numerical residual, rank check, or caller flag replaces coverage.

The caller must prove any elimination denominator nonzero, enumerate all contact supports, filter support feasibility over whole root boxes, bind parameters, and certify the requested readout. No roots may be dropped because they are unstable or have higher energy unless the caller explicitly asks that different question. Unsupported extra shell modes, general Neo-Hooke/Ogden laws and full physical models retain OSAKER.

This standalone module does not change existing solver defaults, import numerical dependencies, choose a branch or provide a full hyperelastic FE backend. Its regression checks target lost coverage, changed inputs, fake uniqueness and singular cases. The method is conventional; no 10x or novelty claim accompanies this patch. Independent review is required before integration.
