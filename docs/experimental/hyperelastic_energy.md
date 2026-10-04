# Experimental hyperelastic energy certificates

`field_engine.experimental.hyperelastic_energy` implements exact-rational
a posteriori bounds for three fixed mathematical models. It is optional and
does not change a solver default or claim physical calibration.

* `bar_certificate(stretches, traction='5/6', body=1)` accepts a uniform P1
  positive-stretch proposal for the unit axial NH continuum. It evaluates
  the continuum stress defect, including variation within elements, and
  returns a guaranteed potential-energy interval.
* `scalar_certificate('sphere', center, radius, load)` uses the thick A=1,
  B=2, mu=1 radially symmetric incompressible NH model. It certifies radius
  error and the energy gap of the exact incompressible reconstruction of
  the supplied radius. Energy is normalized by 4*pi. It does not certify the
  raw mixed FE field or stability against nonspherical deformations.
* `scalar_certificate('arch', ...)` uses the constrained-apex two-bar
  two-bar energy. It has no bending or other transverse modes.
* `verify(result)` recomputes every field from the bound model inputs;
  applications must replay externally supplied certificates before use.

Use exact int, decimal/rational string, or Fraction input. Floats and booleans
are rejected. Numerical solver output may be proposed as a decimal string;
the checker treats that decimal as an exact candidate and bounds its error.
CERTIFIED is a mathematical-model status, never a PHYSICAL status.
OSAKER (Swedish OSÄKER) means that this contract has not proved the claim.

The analytic license is equilibrium existence plus a positive whole-interval
curvature margin. The energy bound is residual²/(2*margin). It is local to the
declared branch. A zero/negative margin, lost bracket, invalid positive-radius
domain or unsupported FULL_3D scope refuses certification. Newton convergence
and a point stiffness eigenvalue do not provide this license.

The implementation and its conditional Lean kernel cover the stated models.
The arch scope is `TWO_BAR_CONSTRAINED_APEX`. Classical validated
duality/root methods match these guarantees; no method novelty or 10x speed
claim is made. A general 3D NH/Ogden FE backend is not provided.
