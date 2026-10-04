"""Continuum complementary-energy certificates on admitted 2D triangle disks.

Finite binary64 inputs define exact real dyadics. Solves are suggestions; the
certificate integrates admissible fields exactly and rounds each result out.
No exact PDE solution, saturation constant, or solver residual is an input.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import math
import numbers
import weakref


def rat(x):
    """Exact rational value of a scalar input; never rounds silently.

    Fractions and integers are taken exactly. Floats (and NumPy floats, Decimal)
    are accepted only when their exact value is a finite binary64 dyadic, so a
    long double, a non-dyadic Decimal or a string cannot silently change the
    geometry, coefficient or source that the certificate refers to.
    """
    if isinstance(x,F):
        return x
    if getattr(x,'ndim',None)==0 and hasattr(x,'item'):
        x = x.item() if type(x).__name__!='longdouble' else x
    if isinstance(x,(bool,str,bytes)) or type(x).__name__=='bool_':
        raise TypeError("numeric scalar required")
    if isinstance(x,numbers.Integral):
        return F(int(x))
    y = float(x)
    if not math.isfinite(y):
        raise ValueError("non-finite scalar")
    r = F.from_float(y)
    ratio = getattr(x, 'as_integer_ratio', None)
    if ratio is None or F(*ratio()) != r:
        raise ValueError("scalar is not exactly a binary64 value; pass a Fraction")
    return r


def down(x):
    y = float(x)
    if not math.isfinite(y):
        raise ValueError("certificate overflow")
    return math.nextafter(y, -math.inf) if F.from_float(y) > x else y


def up(x):
    y = float(x)
    if not math.isfinite(y):
        raise ValueError("certificate overflow")
    return math.nextafter(y, math.inf) if F.from_float(y) < x else y


def orient(a, b, c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def intersects(a, b, c, d):
    if any(max(a[i], b[i]) < min(c[i], d[i]) or
           max(c[i], d[i]) < min(a[i], b[i]) for i in (0, 1)):
        return False
    z = [orient(a,b,c), orient(a,b,d), orient(c,d,a), orient(c,d,b)]
    return z[0]*z[1] <= 0 and z[2]*z[3] <= 0


_ADMITTED = object()
# Identity registry: dataclasses.replace() or a hand-built Mesh is never admitted.
_ADMITTED_MESHES = weakref.WeakSet()


def _require_admitted(mesh):
    if type(mesh) is not Mesh or mesh._admitted is not _ADMITTED or mesh not in _ADMITTED_MESHES:
        raise ValueError("use Mesh.admit before certification")


@dataclass(frozen=True, eq=False)
class Mesh:
    xy: tuple
    triangles: tuple
    boundary: tuple
    _admitted: object = None

    @classmethod
    def admit(cls, xy, triangles):
        """Admit an embedded oriented simplicial disk, including reentrant edges.

        Local manifold links + connected faces + Euler1 + one simple boundary
        + positive oriented faces imply a degree-one planar embedding. Holes
        and multiply connected domains are deliberately outside this API.
        """
        p = tuple(tuple(rat(v) for v in row) for row in xy)
        if not p or any(len(row) != 2 for row in p) or len(set(p)) != len(p):
            raise ValueError("unique 2D coordinates required")
        ts = []
        for row in triangles:
            if len(row) != 3 or any(int(v) != v for v in row):
                raise ValueError("integer triangle indices required")
            t = tuple(int(v) for v in row)
            if min(t)<0 or max(t)>=len(p) or len(set(t))!=3:
                raise ValueError("invalid triangle indices")
            if orient(*(p[v] for v in t)) <= 0:
                raise ValueError("nonpositive triangle orientation")
            ts.append(t)
        ts = tuple(ts)
        if not ts or len(set(frozenset(t) for t in ts)) != len(ts):
            raise ValueError("nonempty distinct faces required")
        edges, links = {}, [[] for _ in p]
        for i,t in enumerate(ts):
            for j in range(3):
                a,b = t[j],t[(j+1)%3]
                edges.setdefault(tuple(sorted((a,b))),[]).append((a,b,i))
                links[a].append((b,t[(j+2)%3]))
        if any(len(e)>2 or (len(e)==2 and e[0][:2]!=e[1][:2][::-1])
               for e in edges.values()):
            raise ValueError("nonmanifold or inconsistent edge")
        be = [e[0][:2] for e in edges.values() if len(e)==1]
        nxt = dict(be)
        if len(nxt)!=len(be) or len(set(nxt.values()))!=len(be) or not be:
            raise ValueError("nonmanifold boundary")
        boundary, cur = [], be[0][0]
        while cur not in boundary:
            boundary.append(cur)
            if cur not in nxt:
                raise ValueError("open boundary")
            cur=nxt[cur]
        if cur!=boundary[0] or len(boundary)!=len(be):
            raise ValueError("one boundary loop required")
        if len(p)-len(edges)+len(ts)!=1:
            raise ValueError("disk topology required")
        # Check each vertex link is a single path (boundary) or cycle (interior).
        bs=set(boundary)
        for v, pairs in enumerate(links):
            adj={}
            for a,b in pairs:
                adj.setdefault(a,set()).add(b);adj.setdefault(b,set()).add(a)
            if not adj:
                raise ValueError("unused vertex")
            deg=[len(a) for a in adj.values()]
            if any(d not in (1,2) for d in deg) or deg.count(1)!=(2 if v in bs else 0):
                raise ValueError("nonmanifold vertex")
            seen=set(); stack=[next(iter(adj))]
            while stack:
                n=stack.pop()
                if n not in seen:
                    seen.add(n); stack.extend(adj[n]-seen)
            if len(seen)!=len(adj):
                raise ValueError("disconnected vertex link")
        # Face adjacency must be connected.
        adj=[[] for _ in ts]
        for e in edges.values():
            if len(e)==2:
                i,j=e[0][2],e[1][2];adj[i].append(j);adj[j].append(i)
        seen=set();stack=[0]
        while stack:
            n=stack.pop()
            if n not in seen:
                seen.add(n);stack.extend(adj[n])
        if len(seen)!=len(ts):
            raise ValueError("disconnected mesh")
        for i,(a,b) in enumerate(be):
            for c,d in be[i+1:]:
                if len({a,b,c,d})==4 and intersects(p[a],p[b],p[c],p[d]):
                    raise ValueError("self-intersecting boundary")
        # Adjacent collinear reversal would otherwise evade intersection test.
        for i,b in enumerate(boundary):
            a,c=boundary[i-1],boundary[(i+1)%len(boundary)]
            if orient(p[a],p[b],p[c])==0 and sum((p[a][j]-p[b][j])*(p[c][j]-p[b][j]) for j in (0,1))>=0:
                raise ValueError("overlapping adjacent boundary edges")
        mesh = cls(p,ts,tuple(boundary),_ADMITTED)
        _ADMITTED_MESHES.add(mesh)
        return mesh


def grad(p, vals):
    det=orient(*p)
    return ((vals[0]*(p[1][1]-p[2][1])+vals[1]*(p[2][1]-p[0][1])+vals[2]*(p[0][1]-p[1][1]))/det,
            (vals[0]*(p[2][0]-p[1][0])+vals[1]*(p[0][0]-p[2][0])+vals[2]*(p[1][0]-p[0][0]))/det)


def norm2(a):
    return sum(x*x for x in a)


def affine_energy(area, nodal):
    """Integral of |affine vector|², exact triangle degree-two moments."""
    s=tuple(sum(v[j] for v in nodal) for j in (0,1))
    return area*(sum(norm2(v) for v in nodal)+norm2(s))/12


def fields(mesh, v, psi, k):
    _require_admitted(mesh)
    v,psi=tuple(map(rat,v)),tuple(map(rat,psi))
    if len(v)!=len(mesh.xy) or len(psi)!=len(mesh.xy):
        raise ValueError("nodal field shape mismatch")
    try:
        len(k)
    except TypeError:
        k=[k]*len(mesh.triangles)
    k=tuple(map(rat,k))
    if len(k)!=len(mesh.triangles) or min(k)<=0:
        raise ValueError("positive trianglewise coefficient required")
    return v,psi,k


def poisson_certificate(mesh,v,psi,source=2.0,k=1.0):
    """Bounds C=int f*u for -div(k grad u)=constant f, u=0.

    gap_upper bounds ||sqrt(k) grad(u-v)||². Its energy-error
    bound includes algebraic error. Geometry is EXACT_REPRESENTED_POLYGON;
    physical/model/representation error is UNKNOWN unless separately supplied.
    """
    v,psi,k=fields(mesh,v,psi,k);f=rat(source)
    if any(v[i]!=0 for i in mesh.boundary):
        raise ValueError("zero Dirichlet trace required")
    lo_sum=F(0);hi_sum=F(0);gap_sum=F(0);local=[];energy_sum=F(0)
    cross_sum=F(0);load_sum=F(0)
    for t,kt in zip(mesh.triangles,k):
        p=tuple(mesh.xy[i] for i in t);a=orient(*p)/2
        gv=grad(p,tuple(v[i] for i in t));gp=grad(p,tuple(psi[i] for i in t))
        q=tuple((-f*x/2+gp[1],-f*y/2-gp[0]) for x,y in p)
        e=a*kt*norm2(gv);load=a*f*sum(v[i] for i in t)/3
        lower=2*load-e;upper=affine_energy(a,q)/kt
        mismatch=tuple((qq[0]-kt*gv[0],qq[1]-kt*gv[1]) for qq in q)
        gap=affine_energy(a,mismatch)/kt
        cross=a*sum(sum(qq[j] for qq in q)*gv[j] for j in (0,1))/3
        if gap!=upper+e-2*cross:
            raise ArithmeticError("element mismatch integral failed")
        cross_sum+=cross;load_sum+=load
        lo_sum+=rat(down(lower));hi_sum+=rat(up(upper));gap_sum+=rat(up(gap))
        energy_sum+=rat(up(e));local.append(up(gap))
    if cross_sum!=load_sum:
        raise ArithmeticError("global source flux identity failed")
    return {'lower':down(lo_sum),'upper':up(hi_sum),'gap_upper':up(gap_sum),
            'primal_energy_upper':up(energy_sum),'local_gap_upper':local,
            'quantity':'poisson_compliance','arithmetic':'EXACT_RATIONAL_ELEMENTS_OUTWARD_DYADIC_SUM',
            'geometry':'EXACT_REPRESENTED_POLYGON','representation_error':'UNKNOWN'}


def resistance_certificate(mesh,v,psi,arcs,k=1.0):
    """Two-electrode resistance R in a disk bounded by four ordered arcs.

    arcs partitions every boundary edge into left(v0), top(psi1), right(v1),
    bottom(psi0), in the admitted CCW loop. Arc lists include both corners.
    k is 2D sheet conductance; R has reciprocal sheet-conductance units.
    """
    v,psi,k=fields(mesh,v,psi,k)
    if set(arcs)!={'left','top','right','bottom'}:
        raise ValueError("four typed boundary arcs required")
    used=[]
    for name in ('bottom','right','top','left'):
        ar=tuple(int(i) for i in arcs[name])
        if len(ar)<2 or len(set(ar))!=len(ar):
            raise ValueError("nonempty simple boundary arc required")
        used.extend(zip(ar,ar[1:]))
    boundary_edges={(mesh.boundary[i],mesh.boundary[(i+1)%len(mesh.boundary)]) for i in range(len(mesh.boundary))}
    if len(used)!=len(set(used)) or set(used)!=boundary_edges:
        raise ValueError("arcs must partition oriented boundary edges")
    for ar,data,target in [('left',v,F(0)),('right',v,F(1)),('bottom',psi,F(0)),('top',psi,F(1))]:
        if any(data[int(i)]!=target for i in arcs[ar]):
            raise ValueError("incorrect electrode or stream trace")
    # Four chains in that cyclic order guarantee one unit of flux.
    for a,b in zip(('bottom','right','top','left'),('right','top','left','bottom')):
        if arcs[a][-1]!=arcs[b][0]:
            raise ValueError("arcs must have the declared cyclic order")
    eu=F(0);du=F(0);cross=F(0)
    for t,kt in zip(mesh.triangles,k):
        p=tuple(mesh.xy[i] for i in t);a=orient(*p)/2
        gv=grad(p,tuple(v[i] for i in t));gp=grad(p,tuple(psi[i] for i in t));q=(gp[1],-gp[0])
        eu+=rat(up(a*kt*norm2(gv)));du+=rat(up(a*norm2(q)/kt))
        cross+=a*sum(gv[i]*q[i] for i in (0,1))
    if cross!=1:
        raise ArithmeticError("unit current identity failed")
    if eu<=0:
        raise ValueError("zero primal energy")
    return {'lower':down(1/eu),'upper':up(du),'quantity':'effective_resistance',
            'arithmetic':'EXACT_RATIONAL_ELEMENTS_OUTWARD_DYADIC_SUM',
            'geometry':'EXACT_REPRESENTED_POLYGON','representation_error':'UNKNOWN'}


_SCOPE = ('quantity','geometry','representation_error','method')


def scalar_readout(cert,scale=1.0):
    """Midpoint readout of an ordered interval. Scope labels are carried along:
    the bound is for the represented polygon unless representation is bounded."""
    s=rat(scale)
    if s<=0:raise ValueError("positive readout scale required")
    lo=s*rat(cert['lower']);hi=s*rat(cert['upper'])
    if lo>hi:raise ValueError('ordered certificate required')
    mid=(lo+hi)/2;m=float(mid)
    flo,fhi=down(lo),up(hi)
    bound=max(abs(rat(m)-rat(flo)),abs(rat(fhi)-rat(m)))
    out={'lower':flo,'upper':fhi,'estimate':m,'absolute_error_upper':up(bound),
         'relative_width_upper':up((hi-lo)/lo) if lo>0 else None}
    out.update({key:cert[key] for key in _SCOPE if key in cert})
    return out


def scalar_error_upper(cert,estimate,scale=1.0):
    """Bound the error of an arbitrary raw solver readout in the same quantity."""
    s=rat(scale);y=rat(estimate)
    if s<=0:raise ValueError('positive readout scale required')
    lo=s*rat(cert['lower']);hi=s*rat(cert['upper'])
    if lo>hi:raise ValueError('ordered certificate required')
    return up(max(abs(y-lo),abs(hi-y)))


def solve_poisson_fields(mesh,source=2.0,k=1.0):
    """Optional SciPy construction of admissible v,psi; no exact solution input.

    The proposed solves do not certify themselves. Pass returned fields to
    poisson_certificate, which accounts for every error in these solves.
    """
    import numpy as np
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    _require_admitted(mesh)
    f=float(rat(source));coeff=np.array([float(x) for x in fields(mesh,[0.]*len(mesh.xy),[0.]*len(mesh.xy),k)[2]])
    xy=np.array(mesh.xy,dtype=float);ts=np.array(mesh.triangles);p=xy[ts]
    det=(p[:,1,0]-p[:,0,0])*(p[:,2,1]-p[:,0,1])-(p[:,1,1]-p[:,0,1])*(p[:,2,0]-p[:,0,0]);a=det/2
    if np.any(det<=0) or not np.all(np.isfinite(det)):
        raise ValueError('geometry cannot be resolved by float solver; use external fields')
    g=np.stack([np.roll(p[:,:,1],-1,axis=1)-np.roll(p[:,:,1],-2,axis=1),
                np.roll(p[:,:,0],-2,axis=1)-np.roll(p[:,:,0],-1,axis=1)],axis=2)/det[:,None,None]
    rows=np.repeat(ts,3,axis=1).ravel();cols=np.tile(ts,(1,3)).ravel()
    def matrix(weight):
        local=(a*weight)[:,None,None]*np.einsum('tik,tjk->tij',g,g)
        return sparse.coo_matrix((local.ravel(),(rows,cols)),shape=(len(xy),len(xy))).tocsr()
    K=matrix(coeff);D=matrix(1/coeff)
    b=np.bincount(ts.ravel(),weights=np.repeat(a*f/3,3),minlength=len(xy))
    cg=np.stack([g[:,:,1],-g[:,:,0]],axis=2);q0=-f*p.mean(axis=1)/2
    bd=np.bincount(ts.ravel(),weights=(-a[:,None]/coeff[:,None]*np.einsum('tk,tik->ti',q0,cg)).ravel(),minlength=len(xy))
    def solve(A,rhs,fixed):
        free=np.setdiff1d(np.arange(len(xy)),fixed);out=np.zeros(len(xy))
        if len(free):out[free]=spsolve(A[free][:,free],rhs[free])
        if not np.all(np.isfinite(out)):raise ValueError('nonfinite proposed fields')
        return out
    return solve(K,b,list(mesh.boundary)),solve(D,bd,[0])


def lift_grid_potential(mesh,axes,field,inside_mask):
    """Lift a successful CUTRAND grid output to this exact polygon's H1 field.

    Native matrix energy/readout is not a continuum certificate. Outside grid
    values are extended as zero; every polygon boundary nodal value is exactly
    zero. Certify the resulting field with an admissible source-balanced stream.
    Geometry equivalence between the polygon and SDF is a separate obligation.
    """
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator
    _require_admitted(mesh)
    field=np.asarray(field,dtype=float);mask=np.asarray(inside_mask,dtype=bool)
    if len(axes)!=2 or field.shape!=mask.shape or field.shape!=tuple(len(ax) for ax in axes):
        raise ValueError('2D grid dimensions must match')
    if not np.all(np.isfinite(field[mask])) or not np.all([np.all(np.isfinite(ax)) for ax in axes]):
        raise ValueError('successful finite native solve required')
    interp=RegularGridInterpolator(axes,np.where(mask,field,0.),bounds_error=True)
    v=interp(np.array(mesh.xy,dtype=float));v[list(mesh.boundary)]=0.
    if not np.all(np.isfinite(v)):raise ValueError('nonfinite lift')
    return v
