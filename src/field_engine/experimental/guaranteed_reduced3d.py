"""Certified convex reduced fields on one exact reference tetrahedral mesh.

Only evaluated admissible combinations are certified. The optimizer is an
untrusted numerical proposal. Cross moments are coefficient-free interval
integrals; the common exact traces/divergence do not depend on conductivity.
"""
from fractions import Fraction as F
from itertools import combinations
import math
from types import MappingProxyType
import numpy as np
from . import guaranteed_scalar3d as c
from . import guaranteed_fastcert3d as fc
from .guaranteed_scalar import rat, down, up


def simplex_dyadic(weights, bits=40):
    """Nonnegative exact dyadic simplex, including an exactly summing last entry."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or not len(w) or not np.all(np.isfinite(w)):
        raise ValueError('finite nonempty weight vector required')
    w = np.maximum(w, 0)
    if not w.sum() > 0:
        raise ValueError('positive total weight required')
    w /= w.sum(); denominator = 1 << bits
    integers = [int(x*denominator) for x in w]
    integers[int(np.argmax(w))] += denominator-sum(integers)
    return tuple(F(x, denominator) for x in integers)


def minimize_simplex(matrix):
    """Small active-set QP. Returned proposal always belongs to the exact simplex.

    A singular/ill-conditioned reduced form cannot invalidate certification:
    this function is solely a proposal generator and falls back to a vertex.
    """
    H = np.asarray(matrix, dtype=float); n = len(H)
    if H.shape != (n, n) or n == 0 or not np.all(np.isfinite(H)):
        raise ValueError('finite square quadratic form required')
    H = (H+H.T)/2
    scale = max(float(np.max(np.abs(H))), np.finfo(float).tiny); H = H/scale
    x = np.zeros(n); x[int(np.argmin(np.diag(H)))] = 1
    active = [int(np.argmax(x))]
    for _ in range(8*n+8):
        grad = H@x; j = int(np.argmin(grad))
        if grad[j] >= x@grad-2e-13:
            return simplex_dyadic(x)
        if j not in active:
            active.append(j)
        for _ in range(n+1):
            A = H[np.ix_(active, active)]; k = len(active)
            K = np.zeros((k+1,k+1)); K[:k,:k] = A; K[:k,k] = 1; K[k,:k] = 1
            rhs = np.r_[np.zeros(k),1.]
            y = np.linalg.lstsq(K,rhs,rcond=1e-14)[0][:k]
            if np.min(y) >= -1e-13:
                x[:] = 0; x[active] = np.maximum(y,0); x /= x.sum(); break
            old = x[active]; bad = y < 0
            theta = min(1., float(np.min(old[bad]/(old[bad]-y[bad]))))
            trial = old+theta*(y-old); x[:] = 0; x[active] = np.maximum(trial,0)
            active = [i for i in active if x[i] > 1e-14]
            x /= x.sum()
    return simplex_dyadic(x)


def _nodal_fields(mesh, potential, flux, degree):
    data, dofs, _, _, viv, _, z, _, _ = fc._fields_fast(mesh,potential,flux,1,degree)
    IV = fc.IV; P,V = data['P'],data['V']
    ziv,_ = fc._from_rationals(z)
    fl = IV(ziv.m[data['fid']]*data['fsg'],ziv.r[data['fid']])
    p = [[P[:,i,j] for j in range(3)] for i in range(4)]
    inv3V = (V*3.).recip()
    q = []
    for i in range(4):
        row = []
        for j in range(3):
            acc = IV(np.zeros(len(mesh.tets)))
            for m in range(4):
                if m != i:
                    acc = acc+fl[:,m]*(p[i][j]-p[m][j])
            row.append(acc*inv3V)
        q.append(row)
    a,b,d = ([p[i][j]-p[0][j] for j in range(3)] for i in (1,2,3))
    bc,ca,ab = fc._cross(b,d),fc._cross(d,a),fc._cross(a,b)
    determinant = a[0]*bc[0]+a[1]*bc[1]+a[2]*bc[2]
    invd = determinant.recip()
    lam = [None,[x*invd for x in bc],[x*invd for x in ca],[x*invd for x in ab]]
    lam[0] = [-(lam[1][j]+lam[2][j]+lam[3][j]) for j in range(3)]
    vals = IV(viv.m[dofs],viv.r[dofs])
    if degree == 1:
        gg = [sum((vals[:,i]*lam[i][j] for i in range(4)), IV(np.zeros(len(mesh.tets)))) for j in range(3)]
        g = [gg]*4
    else:
        g = []
        for l in range(4):
            row = []
            for j in range(3):
                acc = sum((vals[:,i]*float(4*int(i == l)-1)*lam[i][j] for i in range(4)),IV(np.zeros(len(mesh.tets))))
                for e,(i,k) in enumerate(combinations(range(4),2)):
                    if i == l: acc = acc+vals[:,4+e]*4.*lam[k][j]
                    if k == l: acc = acc+vals[:,4+e]*4.*lam[i][j]
                row.append(acc)
            g.append(row)
    return g,q,V


def exact_load(mesh,potential,source=1,degree=2):
    dofs,_,_,_ = c.potential_layout(mesh,degree)
    v = tuple(map(rat,potential)); f=rat(source)
    # Integrate with one integer denominator, rather than normalizing thousands
    # of Fraction additions. Inputs remain the identical exact rationals.
    Dv=math.lcm(*(x.denominator for x in v))
    z=[x.numerator*(Dv//x.denominator) for x in v]
    DV=math.lcm(*(x.denominator for x in mesh.volumes))
    Vz=[x.numerator*(DV//x.denominator) for x in mesh.volumes]
    total=0
    for vol,row in zip(Vz,dofs):
        value=sum(z[i] for i in row) if degree==1 else -sum(z[i] for i in row[:4])+4*sum(z[i] for i in row[4:])
        total+=vol*value
    return F(total,Dv*DV*(4 if degree==1 else 20))*f



def _pair(V,a,b,d,e):
    acc = a[0][d]*b[0][e]
    sa,sb = a[0][d],b[0][e]
    for l in range(1,4):
        acc = acc+a[l][d]*b[l][e]; sa=sa+a[l][d]; sb=sb+b[l][e]
    return V*(acc+sa*sb)/20.


class ReducedCertificate:
    """Reference fields with exact common constraints and interval cross energies.

    Instances retain their admitted source mesh and reconstructable rational
    fields; no caller-supplied moment dictionary is accepted as a certificate.
    """
    def __setattr__(self,name,value):
        if name in ('mesh','source','electrodes','degree','labels','regions','diagonal_only') and hasattr(self,name):
            raise AttributeError('certified reference context is immutable')
        object.__setattr__(self,name,value)

    @property
    def fields(self):
        return self._fields

    def __init__(self, mesh, source=1, electrodes=None, degree=2, region_labels=None,
                 diagonal_only=False):
        c.require_mesh(mesh)
        self.mesh=mesh; self.source=rat(source); self.electrodes=None if electrodes is None else MappingProxyType({k:tuple(v) for k,v in electrodes.items()}); self.degree=degree
        self.labels=tuple(region_labels) if region_labels is not None else (None,)*len(mesh.tets)
        if len(self.labels) != len(mesh.tets): raise ValueError('one label per tet required')
        if any(isinstance(x,float) for x in self.labels):raise ValueError('hashable non-float region labels required')
        self.regions=tuple(sorted(set(self.labels),key=repr))
        self._masks=tuple(np.array([x == r for x in self.labels]) for r in self.regions)
        if not np.all(np.sum(self._masks,axis=0)==1):raise ValueError('region labels must form a complete disjoint partition')
        self._fields=(); self._nodal=[]; self._G=MappingProxyType({});self._Q=MappingProxyType({});self.diagonal_only=diagonal_only

    def add(self, potential, flux, normalize_load=True):
        """Validate exact hypotheses BEFORE retaining any field or cross moments."""
        c.require_mesh(self.mesh)
        v=tuple(map(rat,potential)); z=tuple(map(rat,flux))
        original=fc.certificate_fast(self.mesh,v,z,self.source,1,self.electrodes,self.degree)
        if self.electrodes is None and normalize_load:
            L=exact_load(self.mesh,v,self.source,self.degree)
            if L <= 0: raise ValueError('positive load required to normalize Poisson fields')
            v=tuple(x/L for x in v)
        elif self.electrodes is None:
            if exact_load(self.mesh,v,self.source,self.degree) != 1:
                raise ValueError('Poisson reference fields must have exact load 1')
        g,q,V=_nodal_fields(self.mesh,v,z,self.degree); i=len(self.fields)
        old=self._nodal+[(g,q)];newG=dict(self._G);newQ=dict(self._Q)
        for j,(gj,qj) in enumerate(old):
            for r,mask in enumerate(self._masks):
                for d in range(3):
                    for e in range(3):
                        if self.diagonal_only and d != e: continue
                        G=_pair(V,g,gj,d,e)[mask].sum(0)
                        Q=_pair(V,q,qj,d,e)[mask].sum(0)
                        newG[r,i,j,d,e]=(float(G.m),float(G.r))
                        newQ[r,i,j,d,e]=(float(Q.m),float(Q.r))
                        newG[r,j,i,e,d]=newG[r,i,j,d,e]
                        newQ[r,j,i,e,d]=newQ[r,i,j,d,e]
        self._G=MappingProxyType(newG);self._Q=MappingProxyType(newQ)
        self._fields=self._fields+((v,z),); self._nodal.append((g,q)); return original

    def _forms(self,A,conductivity):
        c.require_mesh(self.mesh)
        if not self.fields: raise ValueError('at least one certified field required')
        A,det,inv=fc._mat(A)
        if self.diagonal_only and any(A[i][j] for i in range(3) for j in range(3) if i != j):
            raise ValueError('this basis retained only diagonal metric moments')
        k={r:rat(conductivity[r]) for r in self.regions} if conductivity is not None else {r:F(1) for r in self.regions}
        if conductivity is not None and set(conductivity) != set(self.regions): raise ValueError('exact region keys required')
        if min(k.values()) <= 0: raise ValueError('positive conductivity required')
        M=[[sum((inv[d][a]*inv[e][a] for a in range(3)),F(0)) for e in range(3)] for d in range(3)]
        N=[[sum((A[a][d]*A[a][e] for a in range(3)),F(0)) for e in range(3)] for d in range(3)]
        n=len(self.fields); E=fc.IV(np.zeros((n,n))); D=fc.IV(np.zeros((n,n)))
        for r,region in enumerate(self.regions):
            for d in range(3):
                for e in range(3):
                    if self.diagonal_only and d != e: continue
                    if not M[d][e] and not N[d][e]: continue
                    def get(T):
                        return fc.IV([[T[r,i,j,d,e][0] for j in range(n)] for i in range(n)],
                                     [[T[r,i,j,d,e][1] for j in range(n)] for i in range(n)])
                    if M[d][e]:
                        w,_=fc._from_rationals([k[region]*M[d][e]]); E=E+get(self._G)*w[0]
                    if N[d][e]:
                        w,_=fc._from_rationals([N[d][e]/k[region]]); D=D+get(self._Q)*w[0]
        return E,D,det

    def bounds(self, A=((1,0,0),(0,1,0),(0,0,1)), conductivity=None,
               alpha=None,beta=None):
        E,D,det=self._forms(A,conductivity)
        alpha=minimize_simplex(E.m) if alpha is None else tuple(map(rat,alpha))
        beta=minimize_simplex(D.m) if beta is None else tuple(map(rat,beta))
        for w in (alpha,beta):
            if len(w) != len(self.fields) or min(w)<0 or sum(w)!=1: raise ValueError('exact nonnegative simplex required')
        def evaluate(T,w):
            W,_=fc._from_rationals(w)
            product=T*fc.IV(W.m[:,None],W.r[:,None])*fc.IV(W.m[None,:],W.r[None,:])
            return product.sum(1).sum(0).exact_bounds()
        elo,ehi=evaluate(E,alpha); dlo,dhi=evaluate(D,beta)
        if ehi <= 0: raise fc.UndecidedEnclosure('positive energy required')
        if self.electrodes is None:
            lo,hi=det/ehi,det*dhi
        else:
            lo,hi=1/(det*ehi),dhi/det
        if lo>hi: raise ArithmeticError('ordered complementary bounds required')
        flo,fhi=down(lo),up(hi)
        return {'lower':flo,'upper':fhi,'relative_width_upper':up((F(fhi)-F(flo))/F(flo)) if flo>0 else None,
                'alpha':alpha,'beta':beta,'n_bases':len(self.fields),'base_mesh_sha256':self.mesh.sha256,
                'quantity':'effective_resistance' if self.electrodes is not None else 'poisson_compliance',
                'method':'CERTIFIED_CONVEX_CROSS_MOMENTS','geometry':'EXACT_CONSTRUCTED_TETRAHEDRAL_DOMAIN',
                'physical_geometry_error':'UNKNOWN'}

    def bounds_material_batch(self, conductivities):
        """Certify a batch of fixed-domain material edits with exact simplex weights.

        Array operations enclose each separate quadratic form. The two-field
        closed-form minimum is an untrusted proposal, just like the scalar QP.
        Positive conductivities must specify exactly the original region keys.
        """
        c.require_mesh(self.mesh)
        n=len(self.fields)
        if not n:raise ValueError('at least one certified field required')
        if not conductivities:return []
        kvals=[]
        for conductivity in conductivities:
            if set(conductivity)!=set(self.regions):raise ValueError('exact region keys required')
            vals=[rat(conductivity[r]) for r in self.regions]
            if min(vals)<=0:raise ValueError('positive conductivities required')
            kvals.extend(vals)
        kiv,_=fc._from_rationals(kvals);nr=len(self.regions);nv=len(conductivities)
        km=fc.IV(kiv.m.reshape(nv,nr),kiv.r.reshape(nv,nr));inverse=km.recip()
        E=fc.IV(np.zeros((nv,n,n)));D=fc.IV(np.zeros((nv,n,n)))
        for region in range(nr):
            G=fc.IV(np.zeros((n,n)));Q=fc.IV(np.zeros((n,n)))
            for d in range(3):
                G=G+fc.IV([[self._G[region,i,j,d,d][0] for j in range(n)] for i in range(n)],
                           [[self._G[region,i,j,d,d][1] for j in range(n)] for i in range(n)])
                Q=Q+fc.IV([[self._Q[region,i,j,d,d][0] for j in range(n)] for i in range(n)],
                           [[self._Q[region,i,j,d,d][1] for j in range(n)] for i in range(n)])
            E=E+G*fc.IV(km.m[:,region,None,None],km.r[:,region,None,None])
            D=D+Q*fc.IV(inverse.m[:,region,None,None],inverse.r[:,region,None,None])
        def proposals(T):
            if n!=2:return [minimize_simplex(h) for h in T.m]
            out=[]
            for h in T.m:
                a,b,cc=h[0,0],(h[0,1]+h[1,0])/2,h[1,1]
                curvature=a-2*b+cc
                theta=min(1.,max(0.,float((cc-b)/curvature))) if curvature>0 else float(a<=cc)
                out.append(simplex_dyadic([theta,1-theta]))
            return out
        alpha,beta=proposals(E),proposals(D)
        def evaluate(T,weights):
            # All weights are exact 40-bit dyadics in [0,1], hence exactly binary64.
            W=np.array(weights,dtype=float)
            energy=T*fc.IV(W[:,:,None])*fc.IV(W[:,None,:])
            return energy.sum(2).sum(1)
        primal=evaluate(E,alpha);dual=evaluate(D,beta);out=[]
        for i in range(nv):
            _,ehi=primal[i].exact_bounds();_,dhi=dual[i].exact_bounds()
            if ehi<=0:raise fc.UndecidedEnclosure('positive energy required')
            lo,hi=1/ehi,dhi
            if lo>hi:raise ArithmeticError('ordered complementary bounds required')
            flo,fhi=down(lo),up(hi)
            out.append({'lower':flo,'upper':fhi,'relative_width_upper':up((F(fhi)-F(flo))/F(flo)) if flo>0 else None,
                        'alpha':alpha[i],'beta':beta[i],'n_bases':n,'base_mesh_sha256':self.mesh.sha256,
                        'quantity':'effective_resistance' if self.electrodes is not None else 'poisson_compliance',
                        'method':'CERTIFIED_CONVEX_MATERIAL_BATCH','geometry':'EXACT_CONSTRUCTED_TETRAHEDRAL_DOMAIN',
                        'physical_geometry_error':'UNKNOWN'})
        return out

    def reconstruct(self,weights,kind):
        w=tuple(map(rat,weights))
        if len(w)!=len(self.fields) or min(w)<0 or sum(w)!=1: raise ValueError('exact simplex required')
        which={'potential':0,'flux':1}[kind]
        return tuple(sum((a*field[which][j] for a,field in zip(w,self.fields)),F(0))
                     for j in range(len(self.fields[0][which])))
