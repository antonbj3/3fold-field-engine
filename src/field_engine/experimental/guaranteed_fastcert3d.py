"""Two-stage verifier and shared affine-family certificates for 3D guarantees.

The guarantee is the one of ``guaranteed_scalar3d.certificate``: for an
admitted mesh, a conforming P1/P2 potential with exact traces and an RT0 flux
with exact element balance, ``lower <= C <= upper`` (Poisson compliance) or
``lower <= R <= upper`` (finite-electrode resistance). Only the *energies*
change arithmetic:

1. Stage one (binary64). Every energy, load and moment is evaluated vectorised
   in IEEE binary64 midpoint-radius interval arithmetic. Each operation adds
   its own rounding radius ``u*|mid|`` (+ an underflow term), every computed
   radius is inflated by ``1+2**-40``, and reductions pay ``gamma_n``. The
   result encloses the exact rational energy; endpoints are combined exactly
   and rounded outward. The interval is therefore never narrower than the
   truth, only wider than the exact certificate by about 1e-13 relative.
2. The conditions that make the inequality a theorem are kept exact and
   cheap: traces are compared exactly, wall/right currents and every element
   balance are checked with integers over a common denominator (no Fraction
   normalisation; balances are compared by integer cross-multiplication).
3. Stage two (exact Fraction) runs only when the caller's monotone decision is
   undecided by the binary64 enclosure (``certify_decision``).

Affine families. With the exact moment tensors G_ij=int k d_i v d_j v and
Q_ij=int q_i q_j/k (enclosed in stage one) the transported fields
v o F^-1 and the (Piola) transported current give, for every map
F(x)=Ax+b with det A>0 and transported electrodes, coefficient and constant
source, a certificate on F(Omega) without a new mesh, solve or integral:

    E(A)=det A tr(A^-1 A^-T G)
    resistance:  D(A)=tr(A Q A^T)/det A,          R in [1/E(A), D(A)]
    Poisson:     D(A)=det A tr(A Q A^T),  C in [2 det A L - E(A), D(A)]

These are the per-variant forms of the committed shear-family polynomials.
``stretch_box_bounds`` closes a whole continuous box of axis stretches.
``CertificateLibrary`` stores verified moments keyed on exact objects and
parameter boxes. An entry is admitted only after at least three distinct
variants matched an independent direct certificate on the mapped mesh, no
check ever failed, and the matched variants identify every moment
coordinate the entry is asked about (see ``affine_design``).
"""
from fractions import Fraction as F
from itertools import combinations
import hashlib
import json
import math
import weakref

import numpy as np

from .guaranteed_scalar import rat, down, up
from . import guaranteed_scalar3d as _g3

U = 2.0**-53                # unit roundoff, binary64 round-to-nearest
_C = 1.0 + 2.0**-40         # inflation of every computed (nonnegative) radius
_E = 2.0**-1020             # covers underflow of products/radius computations
ARITHMETIC_FAST = 'BINARY64_MIDRAD_ENCLOSURE_OUTWARD_EXACT_INTEGER_BALANCE'
MOMENTS_FAST = 'BINARY64_ENCLOSED_TETRA_VOLUME_MOMENTS'


class UndecidedEnclosure(ArithmeticError):
    """Stage one cannot decide; the exact certificate must be computed."""


class IV:
    """Midpoint-radius interval arrays: exact value in [m-r, m+r]."""
    __slots__ = ('m', 'r')

    def __init__(self, m, r=None):
        self.m = np.asarray(m, dtype=np.float64)
        self.r = np.zeros_like(self.m) if r is None else np.asarray(r, dtype=np.float64)

    def __getitem__(self, k):
        return IV(self.m[k], self.r[k])

    @staticmethod
    def _rad(r, m):
        return (r + U*np.abs(m))*_C + _E

    def __add__(self, o):
        o = _iv(o); m = self.m + o.m
        return IV(m, IV._rad(self.r + o.r, m))

    __radd__ = __add__

    def __sub__(self, o):
        o = _iv(o); m = self.m - o.m
        return IV(m, IV._rad(self.r + o.r, m))

    def __rsub__(self, o):
        return _iv(o) - self

    def __neg__(self):
        return IV(-self.m, self.r)

    def __mul__(self, o):
        o = _iv(o); m = self.m*o.m
        return IV(m, IV._rad(np.abs(self.m)*o.r + self.r*np.abs(o.m) + self.r*o.r, m))

    __rmul__ = __mul__

    def recip(self):
        lo = self.m - self.r
        if not np.all(lo > 0):
            raise UndecidedEnclosure('reciprocal of an interval that is not strictly positive')
        # An overflowed denominator would erase input uncertainty from the
        # radius. Reject that arithmetic path so callers use the exact reserve.
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            denominator = self.m*lo
        if not np.all(np.isfinite(denominator)) or not np.all(denominator > 0):
            raise UndecidedEnclosure('reciprocal radius denominator overflow or underflow')
        m = 1.0/self.m
        return IV(m, IV._rad(self.r/denominator, m))

    def __truediv__(self, o):
        return self*_iv(o).recip()

    def sq(self):
        m = self.m*self.m
        return IV(m, IV._rad(2*np.abs(self.m)*self.r + self.r*self.r, m))

    def sum(self, axis):
        n = self.m.shape[axis]
        g = n*U*(1 + 2.0**-30)
        if g > 2.0**-20:
            raise ArithmeticError('reduction too long for the gamma_n bound used')
        m = np.sum(self.m, axis=axis)
        r = (np.sum(self.r, axis=axis) + g*np.sum(np.abs(self.m), axis=axis))*(1 + 2*g)*_C + n*_E
        return IV(m, r)

    def exact_bounds(self):
        """Exact rational [lo, hi] of a scalar enclosure."""
        m, r = float(self.m), float(self.r)
        if not (math.isfinite(m) and math.isfinite(r)):
            raise UndecidedEnclosure('binary64 enclosure overflow or underflow; use the exact certificate')
        return F(m) - F(r), F(m) + F(r)


def _iv(x):
    if isinstance(x, IV):
        return x
    return IV(x)


def _from_rationals(values):
    """Enclose exact rationals/finite binary64 scalars by binary64 mid-radius."""
    vals = [rat(x) for x in values]
    m = np.array([float(x) for x in vals], dtype=np.float64)
    if not np.all(np.isfinite(m)):
        raise ValueError('nonfinite value')
    exact = np.array([F.from_float(float(a)) == x for a, x in zip(m, vals)], dtype=bool)
    r = np.where(exact, 0.0, U*np.abs(m) + _E)
    return IV(m, r), vals


def _potential_input(potential):
    """float64 arrays are exact dyadics; anything else goes through rat()."""
    if isinstance(potential, np.ndarray) and potential.dtype == np.float64 and potential.ndim == 1:
        if not np.all(np.isfinite(potential)):
            raise ValueError('non-finite scalar')
        return IV(potential.copy()), potential
    iv, vals = _from_rationals(potential)
    return iv, vals


# ---------------------------------------------------------------- mesh cache
_MESH_CACHE = weakref.WeakKeyDictionary()


def _mesh_data(mesh, degree):
    _g3.require_mesh(mesh)
    data = _MESH_CACHE.get(mesh)
    if data is None:
        # Edge vectors p_i - p_0 are formed exactly (integers over one common
        # denominator) and rounded once, so a far translation cannot cancel.
        X, D = _g3._integer_coordinates(mesh.points)
        if X is None:
            rel = [[tuple(a - b for a, b in zip(mesh.points[i], mesh.points[t[0]])) for i in t] for t in mesh.tets]
            m = np.array([[[float(x) for x in v] for v in row] for row in rel], dtype=np.float64)
        else:
            m = np.array([[[(X[i][j] - X[t[0]][j])/D for j in range(3)] for i in t] for t in mesh.tets], dtype=np.float64)
        if not np.all(np.isfinite(m)):
            raise ValueError('nonfinite coordinates')
        vol, vols = _from_rationals(mesh.volumes)
        fid = np.array([[fi for fi, _ in row] for row in mesh.tet_faces], dtype=np.int64)
        fsg = np.array([[s for _, s in row] for row in mesh.tet_faces], dtype=np.float64)
        data = {'P': IV(m, U*np.abs(m) + _E), 'V': vol, 'vols': vols, 'fid': fid, 'fsg': fsg, 'layout': {}}
        _MESH_CACHE[mesh] = data
    if degree not in data['layout']:
        dofs, boundary, edge_nodes, nv = _g3.potential_layout(mesh, degree)
        data['layout'][degree] = (np.array(dofs, dtype=np.int64), boundary, edge_nodes, nv)
    return data


def _common_denominator_balance(mesh, z, f, vols):
    """Exact element balance sum_i s_i z_i == -f V for every tet, in integers.

    One lcm over the *distinct* denominators, then integer numerators only;
    no Fraction is normalised per operation.
    """
    dens = {x.denominator for x in z}
    dens.update(f.denominator*v.denominator for v in vols)
    L = math.lcm(*dens)
    mult = {d: L//d for d in dens}
    Z = [x.numerator*mult[x.denominator] for x in z]
    fn = f.numerator
    for (row, v) in zip(mesh.tet_faces, vols):
        lhs = 0
        for fi, s in row:
            lhs += Z[fi] if s > 0 else -Z[fi]
        if lhs != -fn*v.numerator*mult[f.denominator*v.denominator]:
            return False
    return True


def _cross(a, b):
    return [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]]


def _affine_energy(V, nodal):
    """V*(sum_l |w_l|^2 + |sum_l w_l|^2)/20 for nodal[l][d] IV arrays (n,)."""
    acc = None
    for l in range(4):
        for d in range(3):
            t = nodal[l][d].sq(); acc = t if acc is None else acc + t
    for d in range(3):
        s = nodal[0][d] + nodal[1][d] + nodal[2][d] + nodal[3][d]
        acc = acc + s.sq()
    return (V*acc)/IV(np.array(20.0))


def _affine_tensor(V, nodal, i, j):
    acc = None
    for l in range(4):
        t = nodal[l][i]*nodal[l][j]; acc = t if acc is None else acc + t
    si = nodal[0][i] + nodal[1][i] + nodal[2][i] + nodal[3][i]
    sj = nodal[0][j] + nodal[1][j] + nodal[2][j] + nodal[3][j]
    return (V*(acc + si*sj))/IV(np.array(20.0))


def _fields_fast(mesh, potential, face_flux, coefficient, degree):
    data = _mesh_data(mesh, degree)
    dofs, boundary, edge_nodes, nv = data['layout'][degree]
    viv, vexact = _potential_input(potential)
    z = tuple(rat(x) for x in face_flux)
    if len(vexact) != nv or len(z) != len(mesh.faces):
        raise ValueError('potential/face-flux shape mismatch')
    if (isinstance(coefficient, (int, float, F)) or getattr(coefficient, 'ndim', None) == 0
            or hasattr(coefficient, 'as_integer_ratio')):
        k0 = rat(coefficient)
        if k0 <= 0:
            raise ValueError('positive matching tetrahedral coefficient required')
        kiv, _ = _from_rationals([k0]); kiv = IV(np.full(len(mesh.tets), kiv.m[0]), np.full(len(mesh.tets), kiv.r[0]))
        coeff = None; constant = True
    else:
        coeff = tuple(map(rat, coefficient))
        if len(coeff) != len(mesh.tets) or min(coeff) <= 0:
            raise ValueError('positive matching tetrahedral coefficient required')
        kiv, _ = _from_rationals(coeff); constant = len(set(coeff)) == 1
    return data, dofs, boundary, edge_nodes, viv, vexact, z, kiv, constant


def certificate_fast(mesh, potential, face_flux, source=1, coefficient=1, electrodes=None,
                     potential_degree=1, with_energy_moments=False, _return_enclosures=False, region_labels=None):
    """Stage-one certificate: same theorem and checks, binary64 enclosed energies.

    Output keys follow ``guaranteed_scalar3d.certificate``; ``arithmetic`` names
    the enclosure. The interval always contains the exact certificate's interval.
    Raises UndecidedEnclosure when binary64 cannot enclose (over/underflow,
    unresolved orientation or ordering); then use the exact certificate.
    """
    with np.errstate(all='ignore'):
        try:
            return _certificate_fast(mesh, potential, face_flux, source, coefficient, electrodes,
                                     potential_degree, with_energy_moments, _return_enclosures, region_labels)
        except OverflowError as err:      # a rational outside the binary64 range
            raise UndecidedEnclosure('binary64 range exceeded; use the exact certificate') from err


def _certificate_fast(mesh, potential, face_flux, source=1, coefficient=1, electrodes=None,
                      potential_degree=1, with_energy_moments=False, _return_enclosures=False, region_labels=None):
    """Stage-one certificate: same theorem and checks, binary64 enclosed energies.

    Output keys follow ``guaranteed_scalar3d.certificate``; ``arithmetic`` names
    the enclosure. The interval always contains the exact certificate's interval.
    """
    degree = potential_degree
    if degree not in (1, 2):
        raise ValueError('potential degree must be 1 or 2')
    data, dofs, boundary_dofs, edge_nodes, viv, vexact, z, kiv, constant = _fields_fast(
        mesh, potential, face_flux, coefficient, degree)
    f = rat(source)
    # ---- exact admissibility (the hypotheses of the theorem) ----
    if electrodes is None:
        if any(vexact[i] != 0 for i in boundary_dofs):
            raise ValueError('zero Dirichlet trace required')
    else:
        if f:
            raise ValueError('resistance has zero body source')
        e, nodes = _g3._electrodes(mesh, electrodes)
        if degree == 2:
            for name in nodes:
                nodes[name].update(edge_nodes[tuple(sorted(pair))] for face in e[name] for pair in combinations(mesh.faces[face], 2))
        if any(vexact[i] != 0 for i in nodes['left']) or any(vexact[i] != 1 for i in nodes['right']):
            raise ValueError('0/1 electrode traces required')
        if any(z[i] != 0 for i in e['wall']) or sum(z[i] for i in e['right']) != 1:
            raise ValueError('unit right current and insulating wall required')
    if not _common_denominator_balance(mesh, z, f, data['vols']):
        raise ValueError('exact element source balance required')
    # ---- stage one: enclosed energies ----
    P, V = data['P'], data['V']; n = len(mesh.tets)
    ziv, _ = _from_rationals(z)
    fl = IV(ziv.m[data['fid']]*data['fsg'], ziv.r[data['fid']])          # (n,4) signed fluxes
    p = [[P[:, i, j] for j in range(3)] for i in range(4)]
    inv3V = (V*IV(np.array(3.0))).recip()
    q = []
    for i in range(4):
        row = []
        for j in range(3):
            acc = None
            for m in range(4):
                if m == i:
                    continue
                t = fl[:, m]*(p[i][j] - p[m][j]); acc = t if acc is None else acc + t
            row.append(acc*inv3V)
        q.append(row)
    a = [p[1][j] - p[0][j] for j in range(3)]; b = [p[2][j] - p[0][j] for j in range(3)]; cc = [p[3][j] - p[0][j] for j in range(3)]
    bc, ca, ab = _cross(b, cc), _cross(cc, a), _cross(a, b)
    d = a[0]*bc[0] + a[1]*bc[1] + a[2]*bc[2]
    if not np.all(d.m - d.r > 0):
        raise UndecidedEnclosure('orientation enclosure not positive')
    invd = d.recip()
    lam = [None, [bc[j]*invd for j in range(3)], [ca[j]*invd for j in range(3)], [ab[j]*invd for j in range(3)]]
    lam[0] = [-(lam[1][j] + lam[2][j] + lam[3][j]) for j in range(3)]
    vals = IV(viv.m[dofs], viv.r[dofs])
    if degree == 1:
        gg = [None]*3
        for j in range(3):
            gg[j] = vals[:, 0]*lam[0][j] + vals[:, 1]*lam[1][j] + vals[:, 2]*lam[2][j] + vals[:, 3]*lam[3][j]
        g = [gg, gg, gg, gg]
        load_t = (V*(vals[:, 0] + vals[:, 1] + vals[:, 2] + vals[:, 3]))*IV(np.array(0.25))
    else:
        g = []
        for l in range(4):
            row = []
            for j in range(3):
                acc = None
                for i in range(4):
                    t = (vals[:, i]*float(4*int(i == l) - 1))*lam[i][j]; acc = t if acc is None else acc + t
                for e_idx, (i, jj) in enumerate(combinations(range(4), 2)):
                    if i == l:
                        acc = acc + (vals[:, 4 + e_idx]*4.0)*lam[jj][j]
                    if jj == l:
                        acc = acc + (vals[:, 4 + e_idx]*4.0)*lam[i][j]
                row.append(acc)
            g.append(row)
        sv = vals[:, 0] + vals[:, 1] + vals[:, 2] + vals[:, 3]
        se = vals[:, 4] + vals[:, 5] + vals[:, 6] + vals[:, 7] + vals[:, 8] + vals[:, 9]
        load_t = V*(se/IV(np.array(5.0)) - sv/IV(np.array(20.0)))
    fiv, _ = _from_rationals([f]); fiv = IV(fiv.m[0], fiv.r[0])
    load_t = load_t*fiv
    ee_t = kiv*_affine_energy(V, g)
    invk = kiv.recip()
    dd_t = _affine_energy(V, q)*invk
    energy = ee_t.sum(0); dual = dd_t.sum(0); load = load_t.sum(0)
    E_lo, E_hi = energy.exact_bounds(); D_lo, D_hi = dual.exact_bounds(); L_lo, L_hi = load.exact_bounds()
    if electrodes is None:
        lo_b, lo_opt = 2*L_lo - E_hi, 2*L_hi - E_lo
        mis = [[q[l][j] - kiv*g[l][j] for j in range(3)] for l in range(4)]
    else:
        if not E_lo > 0:
            if E_hi <= 0:
                raise ValueError('positive primal conductance energy required')
            raise UndecidedEnclosure('positive primal energy not enclosed; use the exact certificate')
        lo_b, lo_opt = 1/E_hi, 1/E_lo
        e_mid = float((E_lo + E_hi)/2); e_rad = up(max(E_hi - F(e_mid), F(e_mid) - E_lo))
        inv_e = IV(np.array(e_mid), np.array(e_rad)).recip()
        mis = [[q[l][j] - kiv*g[l][j]*inv_e for j in range(3)] for l in range(4)]
    hi_b, hi_opt = D_hi, D_lo
    # Every exact lower lies in [lo_b, lo_opt] and every exact upper in [hi_opt, hi_b].
    # Only lo_b > hi_b proves the exact bounds unordered; lo_b > hi_opt does not
    # (a wide enclosure of the upper bound, e.g. from underflow, gives that).
    if lo_b > hi_b:
        raise ArithmeticError('ordered complementary-energy bounds required')
    gap_t = _affine_energy(V, mis)*invk
    local = (gap_t.m + gap_t.r)*_C
    gap = gap_t.sum(0)
    flo, fhi = down(lo_b), up(hi_b); mid = float((rat(flo) + rat(fhi))/2)
    bound = max(abs(rat(mid) - rat(flo)), abs(rat(fhi) - rat(mid)))
    result = {'lower': flo, 'upper': fhi, 'estimate': mid, 'absolute_error_upper': up(bound),
              'relative_width_upper': up((rat(fhi) - rat(flo))/rat(flo)) if flo > 0 else None,
              'primal_energy_upper': up(E_hi), 'load': float((L_lo + L_hi)/2),
              'gap_upper': up(F(float(gap.m)) + F(float(gap.r))),
              'local_gap_upper': [float(x) for x in local], 'flux_load_identity_exact': None,
              'source_balance_exact': True, 'normal_trace_exact': True,
              'quantity': 'effective_resistance' if electrodes is not None else 'poisson_compliance',
              'constant_coefficient': constant,
              'mesh_sha256': mesh.sha256, 'potential_degree': degree,
              'arithmetic': ARITHMETIC_FAST,
              'geometry': 'EXACT_CONSTRUCTED_TETRAHEDRAL_DOMAIN', 'physical_geometry_error': 'UNKNOWN',
              # The exact certificate's interval contains [inner_lower_upper, inner_upper_lower].
              'inner_lower_upper': up(lo_opt), 'inner_upper_lower': down(hi_opt)}
    if with_energy_moments:
        Gm, Gr, Qm, Qr = (np.zeros((3, 3)) for _ in range(4))
        for i in range(3):
            for j in range(i, 3):
                gt = (kiv*_affine_tensor(V, g, i, j)).sum(0)
                qt = (_affine_tensor(V, q, i, j)*invk).sum(0)
                Gm[i, j] = Gm[j, i] = gt.m; Gr[i, j] = Gr[j, i] = gt.r
                Qm[i, j] = Qm[j, i] = qt.m; Qr[i, j] = Qr[j, i] = qt.r
        result['energy_moments'] = {'primal_mid': Gm.tolist(), 'primal_rad': Gr.tolist(),
                                    'dual_mid': Qm.tolist(), 'dual_rad': Qr.tolist(),
                                    'load_mid': float(load.m), 'load_rad': float(load.r),
                                    'arithmetic': MOMENTS_FAST}
    if region_labels is not None:
        labels = list(region_labels)
        if len(labels) != len(mesh.tets) or any(isinstance(x, float) for x in labels):
            raise ValueError('one hashable non-float region label per tetrahedron required')
        keys = sorted(set(labels), key=repr); lab = np.array([keys.index(x) for x in labels])
        reg = {repr(key): {'primal_no_k': [[None]*3 for _ in range(3)], 'dual_no_k': [[None]*3 for _ in range(3)]} for key in keys}
        masks = [lab == r for r in range(len(keys))]
        for i in range(3):
            for j in range(i, 3):
                tg, tq = _affine_tensor(V, g, i, j), _affine_tensor(V, q, i, j)
                for key, mask in zip(keys, masks):
                    gt, qt = tg[mask].sum(0), tq[mask].sum(0); ent = reg[repr(key)]
                    ent['primal_no_k'][i][j] = ent['primal_no_k'][j][i] = (float(gt.m), float(gt.r))
                    ent['dual_no_k'][i][j] = ent['dual_no_k'][j][i] = (float(qt.m), float(qt.r))
        result['region_moments'] = {'regions': reg, 'labels_sha256': hashlib.sha256(repr(labels).encode()).hexdigest(),
                                    'load_mid': float(load.m), 'load_rad': float(load.r), 'arithmetic': MOMENTS_FAST}
    if _return_enclosures:
        result['_enclosures'] = {'energy': (E_lo, E_hi), 'dual': (D_lo, D_hi), 'load': (L_lo, L_hi)}
    return result


def certify_decision(mesh, potential, face_flux, accept, source=1, coefficient=1, electrodes=None,
                     potential_degree=1, with_energy_moments=False):
    """Two-stage verifier for a monotone decision ``accept(lower, upper)``.

    ``accept`` must be monotone: if it holds for an interval it holds for every
    subinterval (e.g. width <= tol, interval inside a design box, upper < t).
    Stage one decides when the binary64 interval is accepted, or when even the
    innermost interval the exact certificate could return is rejected. Only
    otherwise is the exact Fraction certificate computed (stage two).
    Returns (certificate, accepted, stage).
    """
    try:
        cert = certificate_fast(mesh, potential, face_flux, source, coefficient, electrodes,
                                potential_degree, with_energy_moments)
    except UndecidedEnclosure:
        cert = None
    if cert is None:
        pass
    elif accept(cert['lower'], cert['upper']):
        return cert, True, 'BINARY64'
    else:
        inner = (cert['inner_lower_upper'], cert['inner_upper_lower'])
        if inner[0] <= inner[1] and not accept(*inner):
            return cert, False, 'BINARY64'
    exact = _g3.certificate(mesh, potential, face_flux, source, coefficient, electrodes,
                            potential_degree, with_energy_moments)
    return exact, bool(accept(exact['lower'], exact['upper'])), 'EXACT_FRACTION'


# ------------------------------------------------------------ affine families
def _moment_intervals(cert):
    mom = cert.get('energy_moments')
    if not mom:
        raise ValueError('certificate with energy moments required')
    if mom.get('arithmetic') == MOMENTS_FAST:
        G = [[(F(mom['primal_mid'][i][j]) - F(mom['primal_rad'][i][j]), F(mom['primal_mid'][i][j]) + F(mom['primal_rad'][i][j]))
              for j in range(3)] for i in range(3)]
        Q = [[(F(mom['dual_mid'][i][j]) - F(mom['dual_rad'][i][j]), F(mom['dual_mid'][i][j]) + F(mom['dual_rad'][i][j]))
              for j in range(3)] for i in range(3)]
        L = (F(mom['load_mid']) - F(mom['load_rad']), F(mom['load_mid']) + F(mom['load_rad']))
    elif mom.get('arithmetic') == 'EXACT_RATIONAL_TETRA_VOLUME_MOMENTS':
        G = [[(F(x), F(x)) for x in row] for row in mom['primal']]
        Q = [[(F(x), F(x)) for x in row] for row in mom['dual']]
        L = (F(mom['load_exact']),)*2
    else:
        raise ValueError('unknown moment arithmetic')
    return G, Q, L


def _contract(W, T):
    """Exact enclosure of sum_ij W_ij T_ij with exact W and interval T."""
    lo = hi = F(0)
    for i in range(3):
        for j in range(3):
            w = W[i][j]
            if w >= 0:
                lo += w*T[i][j][0]; hi += w*T[i][j][1]
            else:
                lo += w*T[i][j][1]; hi += w*T[i][j][0]
    return lo, hi


def _mat(A):
    A = tuple(tuple(map(rat, row)) for row in A)
    if len(A) != 3 or any(len(r) != 3 for r in A):
        raise ValueError('3x3 matrix required')
    det = A[0][0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1]) - A[0][1]*(A[1][0]*A[2][2] - A[1][2]*A[2][0]) + A[0][2]*(A[1][0]*A[2][1] - A[1][1]*A[2][0])
    if det <= 0:
        raise ValueError('positive determinant required')
    cof = [[A[1][1]*A[2][2]-A[1][2]*A[2][1], -(A[0][1]*A[2][2]-A[0][2]*A[2][1]), A[0][1]*A[1][2]-A[0][2]*A[1][1]],
           [-(A[1][0]*A[2][2]-A[1][2]*A[2][0]), A[0][0]*A[2][2]-A[0][2]*A[2][0], -(A[0][0]*A[1][2]-A[0][2]*A[1][0])],
           [A[1][0]*A[2][1]-A[1][1]*A[2][0], -(A[0][0]*A[2][1]-A[0][1]*A[2][0]), A[0][0]*A[1][1]-A[0][1]*A[1][0]]]
    inv = [[cof[i][j]/det for j in range(3)] for i in range(3)]
    return A, det, inv


def _poisson_lower(load_lo, energy_hi):
    """max over lambda of 2 lambda F - lambda^2 E for the scaled transported potential.

    lambda*v is admissible for every real lambda (zero trace), so the Dirichlet
    principle gives C >= F^2/E (F > 0); otherwise the lambda=1 value. Exact.
    """
    plain = 2*load_lo - energy_hi
    if load_lo > 0 and energy_hi > 0:
        return max(plain, load_lo*load_lo/energy_hi)
    return plain


def _check_family_cert(cert):
    if cert.get('quantity') not in ('poisson_compliance', 'effective_resistance') or not cert.get('constant_coefficient'):
        raise ValueError('constant-coefficient typed scalar response required')


def affine_variant_bounds(cert, A, rescale_potential=True):
    """Certified interval on F(Omega), F(x)=Ax+b, from the base certificate's moments.

    Valid for every positive-determinant affine map with the same vertex/face
    identities (electrodes, coefficient and constant source transported).
    Exact rational evaluation of a few dozen terms; no mesh, solve or integral.
    """
    _check_family_cert(cert)
    G, Q, L = _moment_intervals(cert)
    A, det, inv = _mat(A)
    M = [[sum(inv[i][k]*inv[j][k] for k in range(3)) for j in range(3)] for i in range(3)]   # A^-1 A^-T
    N = [[sum(A[k][i]*A[k][j] for k in range(3)) for j in range(3)] for i in range(3)]       # A^T A
    e_lo, e_hi = _contract(M, G); e_lo, e_hi = det*e_lo, det*e_hi
    d_lo, d_hi = _contract(N, Q)
    if cert['quantity'] == 'effective_resistance':
        if e_lo <= 0:
            raise ValueError('positive transported primal energy required')
        lo, hi = 1/e_hi, d_hi/det
    else:
        lo, hi = (_poisson_lower(det*L[0], e_hi) if rescale_potential else 2*det*L[0] - e_hi), det*d_hi
    if lo > hi:
        raise ArithmeticError('ordered bounds required')
    flo, fhi = down(lo), up(hi)
    return {'lower': flo, 'upper': fhi,
            'relative_width_upper': up((rat(fhi) - rat(flo))/rat(flo)) if flo > 0 else None,
            'quantity': cert['quantity'], 'base_mesh_sha256': cert['mesh_sha256'],
            'map_det': str(det), 'method': 'TRANSPORTED_FIELDS_EXACT_AFFINE_MOMENT_CONTRACTION',
            '_energy': (e_lo, e_hi), '_load': (det*L[0], det*L[1])}


def material_variant_bounds(cert, conductivity, A=((1, 0, 0), (0, 1, 0), (0, 0, 1)), rescale_potential=True):
    """Certified interval for new region conductivities k_r (and an optional affine map).

    The base fields stay admissible for every positive k (traces and div q do not
    involve k); E = det A sum_r k_r tr(A^-1 A^-T G_r), D = sum_r tr(A Q_r A^T)/k_r
    (/det A for resistance, *det A for Poisson), with G_r, Q_r the coefficient-free
    region moments of the base certificate. Exact evaluation, no solve.
    """
    rm = cert.get('region_moments')
    if not rm or rm.get('arithmetic') != MOMENTS_FAST:
        raise ValueError('certificate with region moments required')
    if cert.get('quantity') not in ('poisson_compliance', 'effective_resistance'):
        raise ValueError('typed scalar response required')
    k = {repr(key): rat(val) for key, val in conductivity.items()}
    if set(k) != set(rm['regions']) or min(k.values()) <= 0:
        raise ValueError('one positive conductivity for exactly the certified regions required')
    A, det, inv = _mat(A)
    M = [[sum(inv[i][kk]*inv[j][kk] for kk in range(3)) for j in range(3)] for i in range(3)]
    N = [[sum(A[kk][i]*A[kk][j] for kk in range(3)) for j in range(3)] for i in range(3)]
    def iv(T):
        return [[(F(m) - F(r), F(m) + F(r)) for m, r in row] for row in T]
    e_lo = e_hi = d_lo = d_hi = F(0)
    for key, mom in rm['regions'].items():
        a, b = _contract(M, iv(mom['primal_no_k'])); e_lo += k[key]*a; e_hi += k[key]*b
        a, b = _contract(N, iv(mom['dual_no_k'])); d_lo += a/k[key]; d_hi += b/k[key]
    e_lo, e_hi = det*e_lo, det*e_hi
    load_lo = F(rm['load_mid']) - F(rm['load_rad']); load_hi = F(rm['load_mid']) + F(rm['load_rad'])
    if cert['quantity'] == 'effective_resistance':
        if e_lo <= 0:
            raise ValueError('positive transported primal energy required')
        lo, hi = 1/e_hi, d_hi/det
    else:
        lo, hi = (_poisson_lower(det*load_lo, e_hi) if rescale_potential else 2*det*load_lo - e_hi), det*d_hi
    if lo > hi:
        raise ArithmeticError('ordered bounds required')
    flo, fhi = down(lo), up(hi)
    return {'lower': flo, 'upper': fhi,
            'relative_width_upper': up((rat(fhi) - rat(flo))/rat(flo)) if flo > 0 else None,
            'quantity': cert['quantity'], 'base_mesh_sha256': cert['mesh_sha256'],
            'method': 'TRANSPORTED_FIELDS_REGION_MATERIAL_AND_AFFINE_MOMENTS',
            '_energy': (e_lo, e_hi), '_load': (det*load_lo, det*load_hi)}


def stretch_box_bounds(cert, box):
    """One certificate for every diagonal stretch diag(s) with s_i in [lo_i, hi_i].

    E(s)=sum_i G_ii s1 s2 s3/s_i^2; each term is monotone in every s_j, so the
    box maximum is bounded by the sum of corner maxima (all exact).
    """
    _check_family_cert(cert)
    G, Q, L = _moment_intervals(cert)
    box = [tuple(map(rat, b)) for b in box]
    if len(box) != 3 or any(not (0 < b[0] <= b[1]) for b in box):
        raise ValueError('three positive ordered stretch intervals required')
    lo_s = [b[0] for b in box]; hi_s = [b[1] for b in box]
    Emax = F(0)
    for i in range(3):
        j, k = [x for x in range(3) if x != i]
        Emax += max(G[i][i][1], F(0))*hi_s[j]*hi_s[k]/lo_s[i]
    Pmin = lo_s[0]*lo_s[1]*lo_s[2]; Pmax = hi_s[0]*hi_s[1]*hi_s[2]
    if cert['quantity'] == 'effective_resistance':
        Dmax = F(0)
        for i in range(3):
            j, k = [x for x in range(3) if x != i]
            Dmax += max(Q[i][i][1], F(0))*hi_s[i]/(lo_s[j]*lo_s[k])
        if Emax <= 0:
            raise ValueError('positive transported primal energy required')
        lo, hi = 1/Emax, Dmax
    else:
        Dmax = sum(max(Q[i][i][1], F(0))*hi_s[i]**2 for i in range(3))*Pmax
        lo = _poisson_lower(min(Pmin*L[0], Pmax*L[0]), Emax); hi = Dmax
    flo, fhi = down(lo), up(hi)
    return {'lower': flo, 'upper': fhi,
            'relative_width_upper': up((rat(fhi) - rat(flo))/rat(flo)) if flo > 0 else None,
            'quantity': cert['quantity'], 'base_mesh_sha256': cert['mesh_sha256'],
            'scope': 'ALL_CONTINUOUS_DIAGONAL_STRETCHES_IN_BOX', 'box': [[str(a), str(b)] for a, b in box]}


# --------------------------------------------------------- certificate book
def _key(mesh, cert, electrodes, coefficient, source, family, box):
    e = None if electrodes is None else {k: sorted(int(i) for i in v) for k, v in sorted(electrodes.items())}
    return hashlib.sha256(json.dumps([mesh.sha256, cert['quantity'], cert['potential_degree'], e,
                                      str(rat(coefficient)), str(rat(source)), family,
                                      [[str(rat(a)), str(rat(b))] for a, b in box]]).encode()).hexdigest()


def _rank(rows):
    """Exact rank of a list of rational rows (Gaussian elimination)."""
    m = [list(map(F, r)) for r in rows]; rank = 0; ncol = len(m[0]) if m else 0
    for col in range(ncol):
        piv = next((i for i in range(rank, len(m)) if m[i][col] != 0), None)
        if piv is None:
            continue
        m[rank], m[piv] = m[piv], m[rank]
        for i in range(len(m)):
            if i != rank and m[i][col] != 0:
                f = m[i][col]/m[rank][col]; m[i] = [a - f*b for a, b in zip(m[i], m[rank])]
        rank += 1
    return rank


def affine_design(A, diagonal=False):
    """Moment coordinates a direct check of map A pins: E(A) and D(A) are linear in
    the symmetric moments G and Q with these coefficient rows (diagonal=True: a
    family of diagonal stretches, where only G_ii and Q_ii enter)."""
    A, det, inv = _mat(A)
    if diagonal and any(A[i][j] for i in range(3) for j in range(3) if i != j):
        raise ValueError('diagonal design requires a diagonal map')
    M = [[sum(inv[i][k]*inv[j][k] for k in range(3)) for j in range(3)] for i in range(3)]
    N = [[sum(A[k][i]*A[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    idx = [(i, i) for i in range(3)] if diagonal else [(i, j) for i in range(3) for j in range(i, 3)]
    w = lambda i, j: 1 if i == j else 2
    return {'E': tuple(det*M[i][j]*w(i, j) for i, j in idx), 'D': tuple(N[i][j]*w(i, j) for i, j in idx)}


def material_design(cert, conductivity):
    """Moment coordinates a direct material check pins (identity map): E = sum_r k_r tr G_r,
    D = sum_r tr Q_r / k_r."""
    rm = cert.get('region_moments')
    if not rm:
        raise ValueError('certificate with region moments required')
    k = {repr(key): rat(val) for key, val in conductivity.items()}
    keys = sorted(rm['regions'])
    if set(k) != set(keys):
        raise ValueError('one conductivity per certified region required')
    return {'E': tuple(k[x] for x in keys), 'D': tuple(1/k[x] for x in keys)}


class CertificateLibrary:
    """Book of verified family certificates keyed on exact objects and boxes.

    Admission: an entry answers only after >=3 distinct variants matched an
    independent direct certificate on the mapped mesh (green), with no
    mismatch ever (one red closes the entry for good; the key cannot be added
    again), after it was needed at least once, and only for variants whose
    moment coordinates lie in the span of the green checks' coordinates
    (``affine_design``/``material_design``). Without that span condition three
    green checks can all agree with a wrong moment tensor (for example three
    isotropic scalings cannot see a swap of two axes).
    """

    def __init__(self):
        self.entries = {}

    def add(self, mesh, cert, electrodes, coefficient, source, family, box):
        if cert.get('mesh_sha256') != mesh.sha256:
            raise ValueError('certificate is not bound to this mesh')
        _g3.require_mesh(mesh)
        key = _key(mesh, cert, electrodes, coefficient, source, family, box)
        if key in self.entries:
            raise ValueError('library key already present; an entry is never replaced or reopened')
        self.entries[key] = {'cert': cert, 'family': family,
                             'box': [tuple(map(rat, b)) for b in box], 'green': set(), 'red': 0, 'needed': 0,
                             'design': {}}
        return key

    def lookup(self, mesh, quantity_cert_template, electrodes, coefficient, source, family, params):
        """Return (key, entry) whose exact box contains params, or (None, None)."""
        p = tuple(map(rat, params))
        for key, ent in self.entries.items():
            c = ent['cert']
            if (ent['family'] == family and c['mesh_sha256'] == mesh.sha256 and ent['red'] == 0
                    and key == _key(mesh, c, electrodes, coefficient, source, family, ent['box'])
                    and all(a <= x <= b for x, (a, b) in zip(p, ent['box']))):
                return key, ent
        return None, None

    def record_check(self, key, variant, matched, design=None):
        ent = self.entries[key]
        if matched:
            v = tuple(map(str, variant)); ent['green'].add(v)
            if design is not None:
                ent['design'][v] = {name: tuple(map(rat, row)) for name, row in design.items()}
        else:
            ent['red'] += 1

    def record_needed(self, key):
        self.entries[key]['needed'] += 1

    def admitted(self, key, design=None):
        """True iff the entry may answer. With ``design`` (the coordinates of the
        variant to be answered) the variant's rows must lie in the span of the
        green rows; without it the green rows must have full column rank."""
        ent = self.entries[key]
        if not (len(ent['green']) >= 3 and ent['red'] == 0 and ent['needed'] >= 1):
            return False
        if set(ent['design']) != ent['green']:
            return False                      # every green check must state what it pinned
        for name in ('E', 'D'):
            rows = [d[name] for d in ent['design'].values()]
            if len({len(r) for r in rows}) != 1:
                return False
            r0 = _rank(rows)
            if design is None:
                if r0 != len(rows[0]):
                    return False
            else:
                row = tuple(map(rat, design[name]))
                if len(row) != len(rows[0]) or _rank(rows + [row]) != r0:
                    return False
        return True


def direct_transport_check(mesh, potential, face_flux, cert, A, source=1, coefficient=1, electrodes=None,
                           potential_degree=1, translation=(0, 0, 0)):
    """Independent check of one variant: build the mapped mesh, transport the
    fields (P1/P2 nodal values unchanged, face fluxes Piola-scaled), certify it
    directly and compare with ``affine_variant_bounds``. True iff the direct
    certified interval and the family interval are mutually consistent: the
    direct enclosure lies within the family one up to binary64 enclosure slack.
    """
    A_, det, _ = _mat(A)
    img = _g3.affine_mesh(mesh, A_, translation)
    scale = 1 if electrodes is not None else det       # Poisson keeps div q=-f: q'=Aq
    z = tuple(rat(x)*scale for x in face_flux)
    direct = certificate_fast(img, potential, z, source, coefficient, electrodes, potential_degree,
                              _return_enclosures=True)
    fam = affine_variant_bounds(cert, A_, rescale_potential=False)   # same fields as the direct certificate
    return _compare_direct(direct, fam), direct, fam


def _compare_direct(direct, fam):
    """Endpoints, energy and load of the direct certificate match the family's."""
    tol = F(1, 2**30)
    enc = direct.pop('_enclosures')
    def close(a, b):
        ma, mb = (a[0] + a[1])/2, (b[0] + b[1])/2
        return abs(ma - mb) <= tol*max(abs(ma), abs(mb))
    return (all(abs(rat(direct[k]) - rat(fam[k])) <= tol*abs(rat(fam[k])) for k in ('lower', 'upper'))
            and close(enc['energy'], fam['_energy']) and close(enc['load'], fam['_load']))


def direct_material_check(mesh, potential, face_flux, cert, conductivity, region_labels, source=0,
                          electrodes=None, potential_degree=1):
    """Independent check of one material variant: certify the same fields directly
    with the per-tetrahedron conductivity of the variant and compare."""
    if hashlib.sha256(repr(list(region_labels)).encode()).hexdigest() != cert.get('region_moments', {}).get('labels_sha256'):
        raise ValueError('region labels differ from the certified ones')
    coeff = [rat(conductivity[x]) for x in region_labels]
    direct = certificate_fast(mesh, potential, face_flux, source, coeff, electrodes, potential_degree,
                              _return_enclosures=True)
    fam = material_variant_bounds(cert, conductivity, rescale_potential=False)
    return _compare_direct(direct, fam), direct, fam
