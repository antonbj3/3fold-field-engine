"""Experimental exact 3D complementary energy on constructively embedded meshes.

Fields are suggestions; balance, traces and degree-two integration are exact.
This certifies the stated continuum model, not physical scan/material accuracy.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import permutations, combinations
from collections import deque
import math
import time
import hashlib
import numbers
import weakref


from .guaranteed_scalar import rat, down, up


def sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), F(0))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def det4(p):
    return dot(sub(p[1], p[0]), cross(sub(p[2], p[0]), sub(p[3], p[0])))


def oriented(points, tet):
    tet = tuple(tet)
    d = det4([points[i] for i in tet])
    if not d:
        raise ValueError('zero volume tetrahedron')
    return tet if d > 0 else (tet[1], tet[0], tet[2], tet[3])


_ADMITTED = object()
_ADMITTED_MESHES = weakref.WeakSet()


@dataclass(frozen=True,eq=False)
class Mesh3D:
    points: tuple
    tets: tuple
    volumes: tuple
    faces: tuple
    incidences: tuple
    tet_faces: tuple
    boundary_nodes: tuple
    provenance: str
    sha256: str
    _admitted: object = None


def _integer_coordinates(points):
    """Common-denominator integer coordinates X=D*p (exact), or None if D is huge."""
    dens = {x.denominator for p in points for x in p}
    D = math.lcm(*dens)
    if D.bit_length() > 4096:
        return None, None
    mult = {d: D//d for d in dens}
    return [tuple(x.numerator*mult[x.denominator] for x in p) for p in points], D


def _isub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def _icross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _idot(a, b):
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]


def _construct(points, tets, provenance):
    """Internal constructor: embedding follows from the calling operation.

    Orientation, volumes and the shared-face side test are exact. They are
    evaluated on integer coordinates over one common denominator (no Fraction
    normalisation per operation); a huge denominator falls back to Fractions.
    The admitted object is identical to the Fraction evaluation.
    """
    points = tuple(tuple(rat(x) for x in p) for p in points)
    if not points or any(len(p) != 3 for p in points) or len(set(points)) != len(points):
        raise ValueError('unique three-dimensional vertices required')
    X, D = _integer_coordinates(points)
    if X is None:
        return _construct_fraction(points, tets, provenance)
    out, volumes, six_d3 = [], [], 6*D**3
    for t in tets:
        t = tuple(t)
        p0, p1, p2, p3 = (X[i] for i in t)
        d = _idot(_isub(p1, p0), _icross(_isub(p2, p0), _isub(p3, p0)))
        if not d:
            raise ValueError('zero volume tetrahedron')
        if d < 0:
            t = (t[1], t[0], t[2], t[3]); d = -d
        out.append(t); volumes.append(F(d, six_d3))
    tets = tuple(out)
    if not tets or len(set(frozenset(t) for t in tets)) != len(tets):
        raise ValueError('nonempty distinct tetrahedra required')
    def same_side(face, x, y):
        a, b, c = (X[i] for i in face)
        n = _icross(_isub(b, a), _isub(c, a))
        return _idot(n, _isub(X[x], a))*_idot(n, _isub(X[y], a)) >= 0
    return _finish(points, tets, tuple(volumes), provenance, same_side)


def _construct_fraction(points, tets, provenance):
    tets = tuple(oriented(points, t) for t in tets)
    if not tets or len(set(frozenset(t) for t in tets)) != len(tets):
        raise ValueError('nonempty distinct tetrahedra required')
    volumes = tuple(det4([points[i] for i in t])/6 for t in tets)
    def same_side(face, x, y):
        a, b, c = (points[i] for i in face)
        n = cross(sub(b, a), sub(c, a))
        return dot(n, sub(points[x], a))*dot(n, sub(points[y], a)) >= 0
    return _finish(points, tets, volumes, provenance, same_side)


def _finish(points, tets, volumes, provenance, same_side):
    face_map, faces, inc, tf = {}, [], [], []
    for ti, t in enumerate(tets):
        row = []
        for j in range(4):
            face = tuple(sorted(t[:j]+t[j+1:]))
            if face not in face_map:
                face_map[face] = len(faces); faces.append(face); inc.append([])
            fi = face_map[face]
            if len(inc[fi]) >= 2:
                raise ValueError('nonmanifold face')
            sign = 1 if not inc[fi] else -1
            if inc[fi]:
                old, oldj, _ = inc[fi][0]
                if same_side(face, t[j], tets[old][oldj]):
                    raise ValueError('tetrahedra on same side of shared face')
            inc[fi].append((ti, j, sign)); row.append((fi, sign))
        tf.append(tuple(row))
    boundary = tuple(sorted({v for face, side in zip(faces, inc) if len(side) == 1 for v in face}))
    signature = hashlib.sha256(repr((points, tets)).encode()).hexdigest()
    mesh=Mesh3D(points, tets, volumes, tuple(faces), tuple(tuple(x) for x in inc),
                tuple(tf), boundary, provenance, signature, _ADMITTED)
    _ADMITTED_MESHES.add(mesh)
    return mesh


def require_mesh(mesh):
    if type(mesh) is not Mesh3D or mesh._admitted is not _ADMITTED or mesh not in _ADMITTED_MESHES:
        raise ValueError('use a constructive mesh factory')


def structured_mesh(n=2, lengths=(1, 1, 1), l_prism=False):
    """Freudenthal triangulation of a box, or an extruded reentrant L.

    The L removes the positive x/y corner from a 2x2x1 bounding box. n is
    the number of equal cells along each unit side. Exact construction proves
    nonoverlap and face conformity; no arbitrary tetrahedral mesh is admitted.
    """
    if int(n) != n or n < 1:
        raise ValueError('positive integer subdivisions required')
    n = int(n); L = tuple(map(rat, lengths))
    if len(L) != 3 or min(L) <= 0:
        raise ValueError('positive box lengths required')
    dims = (2*n, 2*n, n) if l_prism else (n, n, n)
    pts, ids, tets = [], {}, []
    def vertex(v):
        if v not in ids:
            ids[v] = len(pts); pts.append(tuple(L[i]*F(v[i], n) for i in range(3)))
        return ids[v]
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                if l_prism and i >= n and j >= n:
                    continue
                for perm in permutations(range(3)):
                    v = [i, j, k]; tet = [vertex(tuple(v))]
                    for ax in perm:
                        v[ax] += 1; tet.append(vertex(tuple(v)))
                    tets.append(tuple(tet))
    return _construct(pts, tets, 'EXACT_FREUDENTHAL_L_PRISM' if l_prism else 'EXACT_FREUDENTHAL_BOX')


def convex_star_mesh(vertices, triangles, center=(0, 0, 0)):
    """Verify a closed convex triangle boundary, then fan to a strict interior.

    All support inequalities, normal orientations and edge orientations use
    rationals. Euler2, connected faces, connected vertex links and paired edges
    establish a sphere; supporting planes plus a strict center give a convex
    embedded boundary and a nonoverlapping fan. Float hull generation is only
    a proposal. Duplicate/coplanar-extra vertices that break links are refused.
    """
    p = tuple(tuple(map(rat, v)) for v in vertices); c = tuple(map(rat, center))
    if len(c) != 3 or any(len(v) != 3 for v in p) or len(set(p)) != len(p) or c in p:
        raise ValueError('distinct 3D boundary vertices and strict center required')
    edges, ts, links = {}, [], [[] for _ in p]
    for tri in triangles:
        if len(tri) != 3 or any(int(i) != i for i in tri):
            raise ValueError('integer triangle indices required')
        t = tuple(map(int, tri))
        if min(t) < 0 or max(t) >= len(p) or len(set(t)) != 3:
            raise ValueError('invalid triangle')
        a, b, d = (p[i] for i in t); normal = cross(sub(b, a), sub(d, a))
        cc = dot(normal, sub(c, a))
        if cc == 0:
            raise ValueError('center on face plane')
        if cc > 0:
            t = (t[0], t[2], t[1]); normal = tuple(-x for x in normal)
        if any(dot(normal, sub(v, a)) > 0 for v in p):
            raise ValueError('surface not a convex supporting-plane boundary')
        for j in range(3):
            aa, bb = t[j], t[(j+1)%3]
            edges.setdefault(tuple(sorted((aa, bb))), []).append((aa, bb, len(ts)))
            links[aa].append((bb, t[(j+2)%3]))
        ts.append(t)
    if len(set(frozenset(t) for t in ts)) != len(ts):
        raise ValueError('duplicate triangles')
    if any(len(e) != 2 or e[0][:2] != e[1][:2][::-1] for e in edges.values()):
        raise ValueError('closed consistently oriented boundary required')
    if len(p)-len(edges)+len(ts) != 2:
        raise ValueError('sphere topology required')
    for pairs in links:
        adj = {}
        for a, b in pairs:
            adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
        if not adj or any(len(v) != 2 for v in adj.values()):
            raise ValueError('cyclic vertex link required')
        seen, stack = set(), [next(iter(adj))]
        while stack:
            v = stack.pop()
            if v not in seen:
                seen.add(v); stack.extend(adj[v]-seen)
        if len(seen) != len(adj):
            raise ValueError('disconnected vertex link')
    adj = [set() for _ in ts]
    for e in edges.values():
        i, j = e[0][2], e[1][2]; adj[i].add(j); adj[j].add(i)
    seen, stack = set(), [0]
    while stack:
        i = stack.pop()
        if i not in seen:
            seen.add(i); stack.extend(adj[i]-seen)
    if len(seen) != len(ts):
        raise ValueError('connected convex boundary required')
    return _construct(p+(c,), [(len(p),)+t for t in ts], 'EXACT_CONVEX_SUPPORT_FAN')


def bisect_edges(mesh, edges):
    """Split every incident tetrahedron for each requested edge, exactly.

    Identical midpoint and shared face subdivisions preserve the represented
    domain and conformity. Later requested edges already removed are skipped.
    """
    require_mesh(mesh)
    # An insertion-ordered dict reproduces the list order "kept tetrahedra,
    # then the split pairs" of a full list rebuild, with vertex incidence sets
    # instead of a scan of all tetrahedra per edge.
    p = list(mesh.points); alive = {}; by_vertex = {}; key = 0
    for t in mesh.tets:
        alive[key] = t
        for v in t:
            by_vertex.setdefault(v, set()).add(key)
        key += 1
    for edge in sorted(set(tuple(sorted(e)) for e in edges)):
        a, b = edge
        if a == b or min(edge) < 0 or max(edge) >= len(mesh.points):
            raise ValueError('valid original edge required')
        incident = sorted(by_vertex.get(a, set()) & by_vertex.get(b, set()))
        if not incident:
            continue
        mid = tuple((x+y)/2 for x, y in zip(p[a], p[b])); m = len(p); p.append(mid)
        old = [alive.pop(k) for k in incident]
        for k, t in zip(incident, old):
            for v in t:
                by_vertex[v].discard(k)
        for t in old:
            for new in (tuple(m if v == a else v for v in t), tuple(m if v == b else v for v in t)):
                alive[key] = new
                for v in new:
                    by_vertex.setdefault(v, set()).add(key)
                key += 1
    return _construct(p, list(alive.values()), mesh.provenance+';EXACT_CONFORMING_EDGE_BISECTION')


def longest_edges(mesh, indices):
    require_mesh(mesh)
    out = []
    for i in indices:
        edges = list(combinations(mesh.tets[int(i)], 2))
        out.append(max(edges, key=lambda e: (dot(sub(mesh.points[e[0]], mesh.points[e[1]]),
                                                 sub(mesh.points[e[0]], mesh.points[e[1]])), tuple(sorted(e)))))
    return out


def affine_mesh(mesh, matrix, translation=(0,0,0)):
    """Construct the exact image under a positive-determinant affine map.

    An invertible global affine map preserves embedding/conformity. Vertex
    indices and face IDs remain unchanged, so typed electrode traces transfer.
    """
    require_mesh(mesh);A=tuple(tuple(map(rat,row)) for row in matrix);b=tuple(map(rat,translation))
    if len(A)!=3 or any(len(row)!=3 for row in A) or len(b)!=3:
        raise ValueError('finite 3x3 affine matrix and 3-vector translation required')
    determinant=dot(A[0],cross(A[1],A[2]))
    if determinant<=0:
        raise ValueError('positive affine determinant required')
    points=[tuple(dot(row,p)+shift for row,shift in zip(A,b)) for p in mesh.points]
    out=_construct(points,mesh.tets,mesh.provenance+';EXACT_POSITIVE_AFFINE_IMAGE')
    if out.faces!=mesh.faces or out.tet_faces!=mesh.tet_faces:
        raise ArithmeticError('affine electrode-face identity failed')
    return out


def grad(p, values):
    a, b, c = (sub(p[i], p[0]) for i in (1, 2, 3))
    d = dot(a, cross(b, c))
    terms = [(values[1]-values[0], cross(b, c)), (values[2]-values[0], cross(c, a)),
             (values[3]-values[0], cross(a, b))]
    return tuple(sum(v*n[j] for v, n in terms)/d for j in range(3))


def affine_energy(volume, nodal):
    s = tuple(sum(v[j] for v in nodal) for j in range(3))
    return volume*(sum(dot(v, v) for v in nodal)+dot(s, s))/20


def potential_layout(mesh, degree=1):
    require_mesh(mesh)
    if degree == 1:
        return mesh.tets, mesh.boundary_nodes, {}, len(mesh.points)
    if degree != 2:
        raise ValueError('potential degree must be 1 or 2')
    edges = sorted({tuple(sorted(e)) for t in mesh.tets for e in combinations(t,2)})
    edge_nodes = {e: len(mesh.points)+i for i,e in enumerate(edges)}
    boundary = set(mesh.boundary_nodes)
    for face, inc in zip(mesh.faces, mesh.incidences):
        if len(inc) == 1:
            boundary.update(edge_nodes[tuple(sorted(e))] for e in combinations(face,2))
    dofs = tuple(t+tuple(edge_nodes[tuple(sorted(e))] for e in combinations(t,2)) for t in mesh.tets)
    return dofs, tuple(sorted(boundary)), edge_nodes, len(mesh.points)+len(edges)


def potential_gradients(p, values, degree):
    if degree == 1:
        g = grad(p,values)
        return (g,g,g,g)
    gradients = tuple(grad(p,tuple(F(int(i==j)) for j in range(4))) for i in range(4))
    result=[]
    for l in range(4):
        g = [sum(values[i]*(4*int(i==l)-1)*gradients[i][d] for i in range(4)) for d in range(3)]
        for value,(i,j) in zip(values[4:], combinations(range(4),2)):
            for d in range(3):
                g[d] += 4*value*(int(i==l)*gradients[j][d]+int(j==l)*gradients[i][d])
        result.append(tuple(g))
    return tuple(result)


def affine_pairing(volume, a, b):
    sa = tuple(sum(v[j] for v in a) for j in range(3)); sb = tuple(sum(v[j] for v in b) for j in range(3))
    return volume*(sum(dot(x,y) for x,y in zip(a,b))+dot(sa,sb))/20


def affine_tensor(volume,nodal):
    sums=tuple(sum(v[j] for v in nodal) for j in range(3))
    return tuple(tuple(volume*(sum(v[i]*v[j] for v in nodal)+sums[i]*sums[j])/20 for j in range(3)) for i in range(3))


def _fields(mesh, potential, face_flux, coefficient, degree=1):
    require_mesh(mesh)
    v, z = tuple(map(rat, potential)), tuple(map(rat, face_flux))
    if len(v) != potential_layout(mesh,degree)[3] or len(z) != len(mesh.faces):
        raise ValueError('potential/face-flux shape mismatch')
    return v, z, _coefficients(mesh, coefficient)


def _coefficients(mesh, coefficient):
    if isinstance(coefficient, (numbers.Number,F)) or getattr(coefficient,'ndim',None)==0 or hasattr(coefficient,'as_integer_ratio'):
        coefficient = [coefficient]*len(mesh.tets)
    coeff = tuple(map(rat, coefficient))
    if len(coeff) != len(mesh.tets) or min(coeff) <= 0:
        raise ValueError('positive matching tetrahedral coefficient required')
    return coeff


def box_electrodes(mesh, axis=0):
    require_mesh(mesh)
    if axis not in (0, 1, 2):
        raise ValueError('axis must be 0,1,2')
    lo, hi = min(p[axis] for p in mesh.points), max(p[axis] for p in mesh.points)
    result = {'left': [], 'right': [], 'wall': []}
    for i, (face, inc) in enumerate(zip(mesh.faces, mesh.incidences)):
        if len(inc) == 1:
            coords = [mesh.points[v][axis] for v in face]
            name = 'left' if all(x == lo for x in coords) else 'right' if all(x == hi for x in coords) else 'wall'
            result[name].append(i)
    if not result['left'] or not result['right']:
        raise ValueError('two finite electrode patches required')
    return {k: tuple(v) for k, v in result.items()}


def _electrodes(mesh, electrodes):
    if set(electrodes) != {'left', 'right', 'wall'}:
        raise ValueError('typed left/right/wall boundary faces required')
    if any(isinstance(i,(bool,str,bytes)) or type(i).__name__=='bool_' or int(i)!=i for values in electrodes.values() for i in values):
        raise ValueError('exact integer electrode-face identifiers required')
    e = {k: tuple(int(i) for i in values) for k, values in electrodes.items()}
    ids = sum((list(v) for v in e.values()), [])
    boundary = {i for i, inc in enumerate(mesh.incidences) if len(inc) == 1}
    if len(set(ids)) != len(ids) or set(ids) != boundary or not e['left'] or not e['right']:
        raise ValueError('electrodes must partition all boundary faces')
    nodes = {name: {v for i in faces for v in mesh.faces[i]} for name, faces in e.items()}
    if nodes['left'] & nodes['right']:
        raise ValueError('distinct separated electrode traces required')
    return e, nodes


def certificate(mesh, potential, face_flux, source=1, coefficient=1, electrodes=None, potential_degree=1, with_energy_moments=False):
    """Poisson C=int f*u, or finite-electrode resistance with unit current.

    Refuses an approximately balanced flux: conservation is an exact equality.
    For resistance the boundary potential is 0/1, wall flux=0, right current=1.
    Geometry is the exact represented domain; physical uncertainty is separate.
    """
    dofs, boundary_dofs, edge_nodes, _ = potential_layout(mesh,potential_degree)
    v, z, coeff = _fields(mesh, potential, face_flux, coefficient,potential_degree); f = rat(source)
    if electrodes is None:
        if any(v[i] != 0 for i in boundary_dofs):
            raise ValueError('zero Dirichlet trace required')
    else:
        if f:
            raise ValueError('resistance has zero body source')
        e, nodes = _electrodes(mesh, electrodes)
        if potential_degree == 2:
            for name in nodes:
                nodes[name].update(edge_nodes[tuple(sorted(pair))] for face in e[name] for pair in combinations(mesh.faces[face],2))
        if any(v[i] != 0 for i in nodes['left']) or any(v[i] != 1 for i in nodes['right']):
            raise ValueError('0/1 electrode traces required')
        if any(z[i] != 0 for i in e['wall']) or sum(z[i] for i in e['right']) != 1:
            raise ValueError('unit right current and insulating wall required')
    energy, load, dual, cross_total, gap_total = (F(0) for _ in range(5)); local = []
    resistance_terms = []
    primal_tensor=[[F(0) for _ in range(3)] for _ in range(3)];dual_tensor=[[F(0) for _ in range(3)] for _ in range(3)]
    for t, volume, row, k, vdofs in zip(mesh.tets, mesh.volumes, mesh.tet_faces, coeff,dofs):
        p = tuple(mesh.points[i] for i in t); fluxes = tuple(s*z[i] for i, s in row)
        if sum(fluxes) != -f*volume:
            raise ValueError('exact element source balance required')
        # Each RT0 basis (x-p_i)/(3V) has outward face integral1.
        q = tuple(tuple(sum(fluxes[i]*(x[j]-p[i][j]) for i in range(4))/(3*volume)
                        for j in range(3)) for x in p)
        values = tuple(v[i] for i in vdofs); g = potential_gradients(p,values,potential_degree)
        ee = k*affine_energy(volume,g)
        ff = volume*f*(sum(values)/4 if potential_degree == 1 else -sum(values[:4])/20+sum(values[4:])/5)
        dd = affine_energy(volume, q)/k
        if with_energy_moments:
            gt,qt=affine_tensor(volume,g),affine_tensor(volume,q)
            for i in range(3):
                for j in range(3):
                    primal_tensor[i][j]+=k*gt[i][j];dual_tensor[i][j]+=qt[i][j]/k
        cc = affine_pairing(volume,q,g)
        mismatch = tuple(tuple(qq[j]-k*gg[j] for j in range(3)) for qq,gg in zip(q,g))
        gg = affine_energy(volume, mismatch)/k
        if gg != dd+ee-2*cc or gg < 0:
            raise ArithmeticError('exact local hypercircle failed')
        energy += ee; load += ff; dual += dd; cross_total += cc; gap_total += gg; local.append(up(gg))
        if electrodes is not None:
            resistance_terms.append((volume, q, g, k))
    expected = load if electrodes is None else F(1)
    if cross_total != expected or gap_total != dual+energy-2*expected:
        raise ArithmeticError('global flux/load/trace identity failed')
    if electrodes is not None:
        if energy <= 0:
            raise ValueError('positive primal conductance energy required')
        gap_total, local = F(0), []
        for volume, q, g, k in resistance_terms:
            mismatch = tuple(tuple(qq[j]-k*gg[j]/energy for j in range(3)) for qq,gg in zip(q,g))
            gg = affine_energy(volume, mismatch)/k
            gap_total += gg; local.append(up(gg))
        if gap_total != dual-1/energy:
            raise ArithmeticError('unit-current resistance gap identity failed')
    lo = 2*load-energy if electrodes is None else 1/energy
    hi = dual
    if lo > hi:
        raise ArithmeticError('ordered complementary-energy bounds required')
    flo, fhi = down(lo), up(hi); mid = float((rat(flo)+rat(fhi))/2)
    bound = max(abs(rat(mid)-rat(flo)), abs(rat(fhi)-rat(mid)))
    result = {'lower': flo, 'upper': fhi, 'estimate': mid, 'absolute_error_upper': up(bound),
            'relative_width_upper': up((rat(fhi)-rat(flo))/rat(flo)) if flo > 0 else None,
            'primal_energy_upper': up(energy), 'load': float(load), 'gap_upper': up(gap_total),
            'local_gap_upper': local, 'flux_load_identity_exact': True,
            'source_balance_exact': True, 'normal_trace_exact': True,
            'quantity': 'effective_resistance' if electrodes is not None else 'poisson_compliance',
            'constant_coefficient': len(set(coeff)) == 1,
            'mesh_sha256': mesh.sha256,
            'potential_degree': potential_degree,
            'arithmetic': 'EXACT_RATIONAL_TETRAHEDRA_OUTWARD_ENDPOINTS',
            'geometry': 'EXACT_CONSTRUCTED_TETRAHEDRAL_DOMAIN', 'physical_geometry_error': 'UNKNOWN'}
    if with_energy_moments:
        if sum(primal_tensor[i][i] for i in range(3))!=energy or sum(dual_tensor[i][i] for i in range(3))!=dual:
            raise ArithmeticError('exact energy tensor trace identity failed')
        result['energy_moments']={'primal':[[str(x) for x in row] for row in primal_tensor],
                                 'dual':[[str(x) for x in row] for row in dual_tensor],
                                 'load_exact':str(load),'arithmetic':'EXACT_RATIONAL_TETRA_VOLUME_MOMENTS'}
    return result


def repair_flux(mesh, proposed, source=1, electrodes=None):
    """Exact spanning-tree elimination of every conservation defect.

    Non-tree face proposals stay dyadic. Leaf corrections flow to a single
    boundary root. For resistance fix the right integral first, and choose a
    left root; zero wall current remains exact. No approximate balance passes.
    All arithmetic is on integers over one common denominator (exactly the
    same rational result as Fraction arithmetic, without per-step gcd).
    """
    require_mesh(mesh); f = rat(source)
    if type(proposed).__name__ == 'ndarray' and str(proposed.dtype) == 'float64' and proposed.ndim == 1:
        values = proposed.tolist()            # exact binary64 dyadics
        if not all(math.isfinite(x) for x in values):
            raise ValueError('non-finite scalar')
        pairs = [x.as_integer_ratio() for x in values]
    else:
        pairs = [(r.numerator, r.denominator) for r in map(rat, proposed)]
    if len(pairs) != len(mesh.faces):
        raise ValueError('face-flux shape mismatch')
    dens = {d for _, d in pairs}; dens.update(f.denominator*V.denominator for V in mesh.volumes)
    L = math.lcm(*dens); mult = {d: L//d for d in dens}
    Z = [n*mult[d] for n, d in pairs]
    if electrodes is None:
        root_face = next(i for i, inc in enumerate(mesh.incidences) if len(inc) == 1)
    else:
        if f:
            raise ValueError('resistance has zero source')
        e, _ = _electrodes(mesh, electrodes)
        for i in e['wall']:
            Z[i] = 0
        Z[e['right'][0]] += L-sum(Z[i] for i in e['right'])
        root_face = e['left'][0]
    root = mesh.incidences[root_face][0][0]
    adjacency = [[] for _ in mesh.tets]
    for fi, inc in enumerate(mesh.incidences):
        if len(inc) == 2:
            a, b = inc[0][0], inc[1][0]; adjacency[a].append((b, fi)); adjacency[b].append((a, fi))
    parent, order = {root: (None, root_face)}, [root]
    for t in order:
        for child, face in adjacency[t]:
            if child not in parent:
                parent[child] = (t, face); order.append(child)
    if len(order) != len(mesh.tets):
        raise ValueError('connected tetrahedral dual graph required')
    source_int = [f.numerator*V.numerator*mult[f.denominator*V.denominator] for V in mesh.volumes]
    defect = [-sv-sum(Z[fi] if s > 0 else -Z[fi] for fi, s in row) for sv, row in zip(source_int, mesh.tet_faces)]
    sign_in = {}
    for fi, inc in enumerate(mesh.incidences):
        for ti, _, s in inc:
            sign_in[(fi, ti)] = s
    for t in reversed(order):
        pt, face = parent[t]; sign = sign_in[(face, t)]
        change = defect[t] if sign > 0 else -defect[t]; Z[face] += change
        if pt is not None:
            other_sign = sign_in[(face, pt)]
            defect[pt] -= change if other_sign > 0 else -change
    if any(sum(Z[fi] if s > 0 else -Z[fi] for fi, s in row) != -sv for sv, row in zip(source_int, mesh.tet_faces)):
        raise ArithmeticError('tree repair failed')
    return tuple(F(x, L) for x in Z)


def solve_fields(mesh, source=1, coefficient=1, electrodes=None, potential_degree=1):
    """SciPy P1 and constrained minimum RT0 energy; solves suggest fields only."""
    import numpy as np
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    require_mesh(mesh); start = time.perf_counter()
    dofs,boundary_dofs,edge_nodes,nv=potential_layout(mesh,potential_degree)
    coeff = _coefficients(mesh, coefficient)
    k = np.array(coeff, dtype=float); p = np.array(mesh.points, dtype=float); ts = np.array(mesh.tets)
    coords = p[ts]; V = np.array(mesh.volumes, dtype=float); f = float(rat(source))
    mat = coords[:, 1:]-coords[:, :1]; inv = np.linalg.inv(mat)
    g = np.concatenate([-inv.sum(axis=2)[:, None, :], np.swapaxes(inv, 1, 2)], axis=1)
    # g[t,i,:] is gradient lambda_i; exact integration follows independently.
    pdofs=np.array(dofs);nd=pdofs.shape[1]
    if potential_degree == 1:
        ke = V[:, None, None]*k[:, None, None]*np.einsum('tik,tjk->tij', g, g)
        load=np.repeat(V*f/4,4)
    else:
        # Basis gradients at the four vertices determine affine gradients.
        vertex_g = g[:,:,None,:]*(4*np.eye(4)[None,:,:,None]-1)
        edge_g = np.stack([4*(np.eye(4)[i][None,:,None]*g[:,j,None,:]+
                              np.eye(4)[j][None,:,None]*g[:,i,None,:]) for i,j in combinations(range(4),2)],axis=1)
        ng=np.concatenate([vertex_g,edge_g],axis=1);sg=ng.sum(axis=2)
        ke=V[:,None,None]*k[:,None,None]/20*(np.einsum('tiaj,tbaj->tib',ng,ng)+np.einsum('tij,tbj->tib',sg,sg))
        load=(V[:,None]*f*np.array([-1/20]*4+[1/5]*6)[None,:]).ravel()
    rows=np.repeat(pdofs,nd,axis=1).ravel();cols=np.tile(pdofs,(1,nd)).ravel()
    K=sparse.coo_matrix((ke.ravel(),(rows,cols)),shape=(nv,nv)).tocsr()
    b=np.bincount(pdofs.ravel(),weights=load,minlength=nv);v=np.zeros(nv)
    if electrodes is None:
        fixed = np.array(boundary_dofs, dtype=int)
    else:
        e, nodes = _electrodes(mesh, electrodes)
        if potential_degree == 2:
            for name in nodes:
                nodes[name].update(edge_nodes[tuple(sorted(pair))] for face in e[name] for pair in combinations(mesh.faces[face],2))
        if f:
            raise ValueError('resistance has zero source')
        v[list(nodes['right'])] = 1; fixed = np.array(sorted(nodes['left']|nodes['right']), dtype=int)
    free = np.setdiff1d(np.arange(nv), fixed)
    assembly_primal = time.perf_counter()-start; t0 = time.perf_counter()
    if len(free):
        v[free] = spsolve(K[free][:, free], (b-K@v)[free])
    primal_s = time.perf_counter()-t0; t0 = time.perf_counter()
    ids = np.array([[fi for fi, _ in row] for row in mesh.tet_faces]); signs = np.array([[s for _, s in row] for row in mesh.tet_faces])
    phi = (coords[:, None, :, :]-coords[:, :, None, :])/(3*V[:, None, None, None])
    sp = phi.sum(axis=2)
    me = V[:, None, None]/(20*k[:, None, None])*(np.einsum('tiaj,tbaj->tib', phi, phi)+np.einsum('tij,tbj->tib', sp, sp))
    me *= signs[:, :, None]*signs[:, None, :]
    M = sparse.coo_matrix((me.ravel(), (np.repeat(ids, 4, axis=1).ravel(), np.tile(ids, (1, 4)).ravel())),
                          shape=(len(mesh.faces), len(mesh.faces))).tocsr()
    B = sparse.coo_matrix((signs.ravel(), (np.repeat(np.arange(len(ts)), 4), ids.ravel())),
                          shape=(len(ts), len(mesh.faces))).tocsr()
    rhs = -f*V
    if electrodes is not None:
        face_free = np.array(sorted(set(range(len(mesh.faces)))-set(e['wall'])), dtype=int)
        constraint = sparse.coo_matrix((np.ones(len(e['right'])), (np.zeros(len(e['right']), dtype=int), np.array(e['right']))),
                                       shape=(1, len(mesh.faces))).tocsr()
        B = sparse.vstack([B, constraint]).tocsr(); rhs = np.r_[rhs, 1.]
    else:
        face_free = np.arange(len(mesh.faces))
    Mf = M[face_free][:, face_free]; Bf = B[:, face_free]
    system = sparse.bmat([[Mf, Bf.T], [Bf, None]], format='csc')
    assembly_dual = time.perf_counter()-t0; t0 = time.perf_counter()
    solution = spsolve(system, np.r_[np.zeros(len(face_free)), rhs]); z = np.zeros(len(mesh.faces)); z[face_free] = solution[:len(face_free)]
    dual_s = time.perf_counter()-t0
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(z)):
        raise ValueError('nonfinite proposed solver fields')
    residual = float(np.max(np.abs(B@z-rhs))); t0 = time.perf_counter()
    exact_z = repair_flux(mesh, z, source, electrodes)
    repair_s = time.perf_counter()-t0
    return v, exact_z, {'assembly_s': assembly_primal+assembly_dual, 'primal_s': primal_s,
                        'dual_s': dual_s, 'balance_repair_s': repair_s, 'proposed_balance_max': residual,
                        'repair_max': max(float(abs(rat(a)-b)) for a, b in zip(z, exact_z))}


def raw_error_upper(cert, readout):
    y = rat(readout)
    lo,hi=rat(cert['lower']),rat(cert['upper'])
    if lo>hi:raise ValueError('ordered modeled-quantity certificate required')
    return up(max(abs(y-lo), abs(y-hi)))


def lift_grid_potential(mesh, axes, field, inside_mask):
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator
    require_mesh(mesh); a = np.asarray(field, dtype=float); mask = np.asarray(inside_mask, dtype=bool)
    if len(axes) != 3 or a.shape != mask.shape or a.shape != tuple(len(x) for x in axes):
        raise ValueError('matching 3D axes/field/mask required')
    if not np.all(np.isfinite(a[mask])) or not all(np.all(np.isfinite(ax)) for ax in axes):
        raise ValueError('successful finite native field required')
    out = RegularGridInterpolator(axes, np.where(mask, a, 0), bounds_error=True)(np.array(mesh.points, dtype=float))
    out[list(mesh.boundary_nodes)] = 0
    return out
