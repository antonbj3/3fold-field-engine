# Homogeneous incompressible solid boundary identity

`homogeneous_boundary.boundary_certificate` evaluates exact polynomial
second variations on a rectangular rod or a parabolic extruded arch.
All parameters and coefficients are integers, rational strings or Fractions.
The implementation checks tangent incompressibility and fixed end traces,
and computes the prestress term by independent volume and face integrals.
Negative admissible energy certifies instability. Positive witness energy
returns `OSAKER` while the infinite boundary spectrum is unverified.

For constant F=diag(t²,t⁻¹,t⁻¹), p=μ/t² and w=F⁻¹v, div w=0 implies
tr((∇w)²)=div((w·∇)w). Every solenoidal zero-full-trace mode therefore
has a(v,v)=μ∫|∇v|², on the entire infinite-dimensional interior kernel.
Orthogonal decomposition in the gradient norm reduces the sign question
to a Stokes harmonic trace lift. No mixed-boundary Korn/Stokes constant,
Lehmann–Goerisch flux enclosure or remaining trace spectrum is certified
by this module. `bulk_kernel_gradient_floor` is explicitly scoped to the
zero-full-trace kernel; it cannot be used as a full-body energy constant.

Rod geometry uses length, width and depth. Both complete end faces are
fixed; lateral tractions are zero in the homogeneous state.
Arch geometry is −1<x<1, 1−x²<y<1−x²+thickness and |z|<depth/2.
Its ends x=±1 are fixed. The other surfaces receive manufactured dead
reference Piola tractions PN with P=μF−pF⁻ᵀ. This loading is a distinct
specified solid test, rather than a crown-loaded physical arch benchmark.
The affine state has exact equilibrium and J=1 in both domains.

The guarantee concerns a constitutive continuum model. Independent
material measurement, fixture uncertainty and variable F/p require
additional analysis. This experimental API has not been integrated.
