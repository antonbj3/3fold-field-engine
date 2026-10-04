# Experimental polynomial 3D elasticity certificates

`field_engine.experimental.guaranteed_elasticity3d` verifies rational global
polynomial fields on the unit cube with homogeneous displacement on all faces,
homogeneous isotropic linear elasticity and polynomial volume forces. It checks
all three equilibrium identities, all six exact traces and exact symmetry.
Material inputs are numeric values or `Fraction`; floats denote their exact
dyadic values. Strings, NaN, infinity, E<=0 and nu outside (-1,1/2) are rejected.

`ElasticCertificate(displacement, stress, body)` constructs its own immutable
moments. Each polynomial is a dict or iterable of `((i,j,k), coefficient)`;
vectors have three components and stress is a full 3x3 tensor. It accepts no
externally supplied moments. `reconstruct_stress` adds symmetric diagonal
antiderivatives of the residual to a proposed deviatoric/pressure stress.
This repair generally changes boundary tractions, so it is admitted only for
the stated all-Dirichlet regime. It is not Arnold-Winther or a patch FE solver.

`exact(E,nu)` and `fast(E,nu)` bound continuum compliance and the squared
displacement energy error. With C=L(u)=a(u,u), A=a(v,v), D=int sigma:S:sigma,

    max(0, 2L(v)-A) <= C <= D
    ||u-v||_a^2 <= A-2L(v)+D
    -D/2 <= min total potential energy <= A/2-L(v)

`decision(E,nu,relative_width=Fraction(1,50))` returns ACCEPT only when the
relative width is at most 2%; otherwise it invokes the exact reserve and
returns OSÄKER if field error still prevents acceptance. The exact reserve also
handles binary64 modulus overflow when the final certificate is representable.
If the final energy itself cannot be represented as finite binary64 endpoints,
the API raises an arithmetic/value error and supplies no false interval.

The compliance tensor is evaluated as

    D = ||dev sigma||_L2^2/(2mu) + ||tr sigma||_L2^2/(9kappa)

rather than a cancellation-prone difference of hydrostatic terms. A fixed
primal divergence defect still contributes kappa*||div v||^2. Uniform bounds
therefore require suitable kinematic reconstruction; stable mixed FE alone
does not make the full displacement energy uniformly small. The published
mixed energy norm with an independent pressure error is a different quantity
when a discrete pressure/divergence relation holds only weakly.

`material_box(E_box, nu_box, reference=(E0,nu0))` bounds the entire continuous
box using monotonicity of material coefficients and optional Loewner reach.
It intersects the two sound bounds directly. Box width includes physical
material variation; it is not a numerical error bar at one material value.
Load, geometry and Dirichlet data must remain fixed. There is no corner license
for arbitrary elastic energy as a function of all problem parameters.

`admit_request` marks unsupported geometry, mixed boundaries, point stress,
contact or exact incompressibility OSÄKER. An energy norm never by itself
certifies a point stress maximum. Lamé/Kirsch curved domains, Hertz contact and
Cook's mixed boundary problem need additional certified adapters.

Comparison with conventional exact controls found equal guarantee capability; no tenfold speedup is claimed.
125 arithmetic enclosures passed a manufactured 3D polynomial test family;
physical PDL validity and general 3D FE reconstruction remain unestablished.
This code is an experimental, separately reviewable port, not a claim of 10x
speed or new method. It depends on the reviewed f94d2aa arithmetic, also present
on 5edc41b. The focused tests include premise perturbations and a no-op gate.
