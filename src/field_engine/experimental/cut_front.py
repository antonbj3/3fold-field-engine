"""Conditional mode III edge slit on a fixed square background grid.

The slit is {phi=y-yc=0, chi=x-a<=0}. It has two traces, zero volume,
and exact rational integration subtriangles. This is also a conforming P1
space: no novelty relative to that control is implied. Solves are proposals;
only feasible trial fields with exactly integrated energies certify the PDE.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import time
import weakref
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from field_engine.experimental.guaranteed_scalar import rat, grad, norm2, orient, down, up

__all__ = ['Slit', 'Moments', 'make_slit', 'assemble', 'solve_fields',
           'exact_moments', 'energy_bounds', 'release_bounds', 'evaluate']

_meshes = weakref.WeakSet()
_moments = weakref.WeakSet()


@dataclass(frozen=True, eq=False)
class Slit:
    xy: tuple
    triangles: tuple
    nx: int
    ny: int
    a: F
    yc: F
    theta: F | None
    groups: tuple
    top: tuple
    bottom: tuple
    insulating_left: tuple
    right: tuple
    area: F
    aggregation: str


@dataclass(frozen=True, eq=False)
class Moments:
    mesh: Slit
    primal: tuple
    dual: tuple


def make_slit(nx, a, *, yc=F(0), aggregate=True, eta=F(1,1000)):
    """Construct a slit disk with coincident, separately indexed crack faces.

    Only this immutable constructive family is admitted. Physical lengths
    are L=4,H=1. Ny=Nx/4: nx denotes cells along the length, not the height.
    All mesh coordinates and area are rational. No arbitrary mesh admission.
    """
    if not isinstance(nx,int) or isinstance(nx,bool) or nx<8 or nx%4:
        raise ValueError('nx must be an integer >=8 divisible by four')
    a,yc,eta=rat(a),rat(yc),rat(eta)
    if not 0<=a<4 or not -F(1,2)<yc<F(1,2) or not 0<eta<F(1,2):
        raise ValueError('finite edge crack and valid aggregation threshold required')
    ny=nx//4; h=F(4,nx); row=(yc+F(1,2))/h
    if row.denominator!=1:
        raise ValueError('yc must lie on a background row; general SDF is UNKNOWN')
    row=int(row)
    xy=[(i*h,-F(1,2)+j*h) for j in range(ny+1) for i in range(nx+1)]
    node=lambda i,j:j*(nx+1)+i
    lower={}
    for i in range(nx+1):
        if i*h<a:
            k=node(i,row); lower[k]=len(xy);xy.append(xy[k])
    cell=a/h; ix=int(cell); theta=cell-ix
    tip=None
    if theta:
        tip=len(xy);xy.append((a,yc))
    triangles=[]
    for j in range(ny):
        for i in range(nx):
            bl,br,tr,tl=node(i,j),node(i+1,j),node(i+1,j+1),node(i,j+1)
            ts=[(bl,br,tr),(bl,tr,tl)]
            if tip is not None and i==ix and j==row:
                ts=[(bl,tip,tr),(tip,br,tr),(bl,tr,tl)]
            elif tip is not None and i==ix and j==row-1:
                ts=[(bl,br,tr),(bl,tr,tip),(bl,tip,tl)]
            if j<row:
                ts=[tuple(lower.get(k,k) for k in t) for t in ts]
            triangles.extend(ts)
    parent=list(range(len(xy)));aggregation='NONE'
    if aggregate and tip is not None:
        if theta<eta:
            master=node(ix,row)
            parent[tip]=master; parent[lower[master]]=master
            aggregation='LEFT_TRACES_AND_TIP'
        elif 1-theta<eta:
            parent[tip]=node(ix+1,row)
            aggregation='TIP_AND_RIGHT_TRACE'
    ids={};groups=[]
    for k in parent:
        ids.setdefault(k,len(ids));groups.append(ids[k])
    top=tuple(node(i,ny) for i in range(nx+1))
    bottom=tuple(node(i,0) for i in range(nx+1))
    # psi is constant on the full left insulating boundary, including BOTH
    # crack traces and the shared tip. Right is its other constant component.
    ins=set(node(0,j) for j in range(ny+1))|set(lower)|set(lower.values())
    if a==0: ins.add(node(0,row))
    if tip is not None: ins.add(tip)
    else: ins.add(node(ix,row))
    right=tuple(node(nx,j) for j in range(ny+1))
    area=sum((orient(*(xy[k] for k in t))/2 for t in triangles),F(0))
    if area!=4 or any(orient(*(xy[k] for k in t))<=0 for t in triangles):
        raise ArithmeticError('constructive partition failed')
    m=Slit(tuple(xy),tuple(triangles),nx,ny,a,yc,theta or None,
           tuple(groups),top,bottom,tuple(sorted(ins)),right,area,aggregation)
    _meshes.add(m)
    return m


def require_mesh(m):
    if type(m) is not Slit or m not in _meshes:
        raise ValueError('certificate requires an unmodified constructive slit')


def geometry(m):
    require_mesh(m)
    xy=np.asarray(m.xy,dtype=float);ts=np.asarray(m.triangles,dtype=int)
    p=xy[ts];b=p[:,1]-p[:,0];c=p[:,2]-p[:,0]
    det=b[:,0]*c[:,1]-b[:,1]*c[:,0]
    if np.any(det<=0) or not np.all(np.isfinite(det)):
        raise ValueError('cut location is unresolved in binary64 assembly; exact clipping never silently snaps')
    g=np.stack((np.stack((p[:,1,1]-p[:,2,1],p[:,2,0]-p[:,1,0]),1),
                np.stack((p[:,2,1]-p[:,0,1],p[:,0,0]-p[:,2,0]),1),
                np.stack((p[:,0,1]-p[:,1,1],p[:,1,0]-p[:,0,0]),1)),1)/det[:,None,None]
    return ts,det/2,g


def region_ids(m):
    # Three synthetic, exactly aligned material regions.
    mids=[sum(m.xy[k][1] for k in t)/3 for t in m.triangles]
    for cut in (-F(1,8),F(1,8)):
        if ((cut+F(1,2))*m.nx/4).denominator!=1:
            raise ValueError('material interface not aligned: region closure UNKNOWN')
    return np.array([0 if y<-F(1,8) else (1 if y<F(1,8) else 2) for y in mids])


def assemble(m, coefficients=1., *, independent=False):
    """Exact gradient grouping before float assembly prevents 1/theta cancellation."""
    ts,areas,g=geometry(m);groups=np.array(m.groups);ng=max(m.groups)+1
    k=np.broadcast_to(np.asarray(coefficients,dtype=float),(len(ts),))
    if not np.all(np.isfinite(k)) or np.any(k<=0):raise ValueError('positive finite mu required')
    if independent:
        rows=[];cols=[];vals=[]
        for t,area,kt in zip(m.triangles,areas,k):
            p=np.array([m.xy[i] for i in t],dtype=float)
            # Independent shape functions from the affine interpolation matrix.
            inv=np.linalg.inv(np.column_stack((np.ones(3),p)))
            exact_g=None
            if len({m.groups[i] for i in t})<3:
                exact_g=[grad(tuple(m.xy[i] for i in t),tuple(F(int(i==j)) for i in range(3))) for j in range(3)]
            gg={}
            for j,i in enumerate(t):
                z=m.groups[i]
                if exact_g is None:gg[z]=gg.get(z,np.zeros(2))+inv[1:,j]
                else:gg[z]=tuple(x+y for x,y in zip(gg.get(z,(F(0),F(0))),exact_g[j]))
            for i,gi in gg.items():
                for j,gj in gg.items():
                    rows.append(i);cols.append(j);vals.append(area*kt*np.dot(np.asarray(gi,float),np.asarray(gj,float)))
        return sparse.coo_matrix((vals,(rows,cols)),shape=(ng,ng)).tocsr()
    mapped=groups[ts];local=areas[:,None,None]*k[:,None,None]*np.einsum('tik,tjk->tij',g,g)
    good=np.array([len(set(r))==3 for r in mapped])
    rows=np.broadcast_to(mapped[good,:,None],local[good].shape).ravel().tolist()
    cols=np.broadcast_to(mapped[good,None,:],local[good].shape).ravel().tolist()
    vals=local[good].ravel().tolist()
    for t,kt in zip(ts[~good],k[~good]):
        pts=tuple(m.xy[i] for i in t);ar=orient(*pts)/2;gg={}
        for j,i in enumerate(t):
            z=m.groups[i];gj=grad(pts,tuple(F(int(j==l)) for l in range(3)))
            gg[z]=tuple(x+y for x,y in zip(gg.get(z,(F(0),F(0))),gj))
        for i,gi in gg.items():
            for j,gj in gg.items():
                rows.append(i);cols.append(j);vals.append(float(ar*rat(float(kt))*sum(x*y for x,y in zip(gi,gj))))
    return sparse.coo_matrix((vals,(rows,cols)),shape=(ng,ng)).tocsr()


def solve_dirichlet(K,groups,zero,one):
    z=set(groups[i] for i in zero);o=set(groups[i] for i in one)
    if z&o:raise ValueError('aggregation conflicts with Dirichlet data')
    fixed=np.array(sorted(z|o),int);free=np.array(sorted(set(range(K.shape[0]))-z-o),int)
    u=np.zeros(K.shape[0]);u[list(o)]=1.
    u[free]=spsolve(K[free][:,free],-K[free][:,fixed]@u[fixed])
    if not np.all(np.isfinite(u)):raise ValueError('failed field proposal')
    return u[np.array(groups)],free,u


def solve_fields(m, mu=(1.,1.,1.), *, independent=False):
    start=time.perf_counter();r=region_ids(m);mu=tuple(map(rat,mu))
    if len(mu)!=3 or min(mu)<=0:raise ValueError('three positive shear moduli required')
    k=np.array([float(mu[i]) for i in r]);K=assemble(m,k,independent=independent)
    KD=assemble(m,1/k,independent=independent);assembled=time.perf_counter()
    v,free,reduced=solve_dirichlet(K,m.groups,m.bottom,m.top)
    psi,_,_=solve_dirichlet(KD,m.groups,m.insulating_left,m.right)
    finish=time.perf_counter()
    reaction=float(reduced@K@reduced)
    return v,psi,{'assembly_s':assembled-start,'solve_s':finish-assembled,
                 'primal_conductance':reaction,'free_dofs':len(free),
                 'algebraic_residual_inf':float(np.max(np.abs((K@reduced)[free])))}


def exact_moments(m,v,psi):
    """Continuum feasibility and exact triangle integrals; no solver assumptions."""
    require_mesh(m);v=tuple(map(rat,v));psi=tuple(map(rat,psi))
    if len(v)!=len(m.xy) or len(psi)!=len(m.xy):raise ValueError('nodal shape mismatch')
    for field,zero,one in ((v,m.bottom,m.top),(psi,m.insulating_left,m.right)):
        if any(field[i]!=0 for i in zero) or any(field[i]!=1 for i in one):
            raise ValueError('inadmissible exact boundary trace')
    pr=[F(0)]*3;du=[F(0)]*3;cross=F(0);r=region_ids(m)
    for t,reg in zip(m.triangles,r):
        p=tuple(m.xy[i] for i in t);area=orient(*p)/2
        gv=grad(p,tuple(v[i] for i in t));gp=grad(p,tuple(psi[i] for i in t));q=(-gp[1],gp[0])
        pr[reg]+=area*norm2(gv);du[reg]+=area*norm2(q)
        cross+=area*sum(x*y for x,y in zip(gv,q))
    if cross!=1:raise ArithmeticError('global exact unit current identity failed')
    moments=Moments(m,tuple(pr),tuple(du));_moments.add(moments)
    return moments


def energy_bounds(moments,mu=(1.,1.,1.),delta=1.):
    if type(moments) is not Moments or moments not in _moments:
        raise ValueError('admitted exact field moments required')
    require_mesh(moments.mesh)
    pr,du=moments.primal,moments.dual;mu=tuple(map(rat,mu));delta=rat(delta)
    if len(mu)!=3 or min(mu)<=0 or delta<=0:raise ValueError('positive mu and displacement required')
    E=sum((k*p for k,p in zip(mu,pr)),F(0));D=sum((q/k for q,k in zip(du,mu)),F(0))
    if E<=0 or D<=0 or E*D<1:raise ArithmeticError('invalid feasible energy ordering')
    lo=delta**2/(2*D);hi=delta**2*E/2
    return {'lower':down(lo),'upper':up(hi),'resistance_lower':down(1/E),
            'resistance_upper':up(D),'reaction_lower':down(delta/D),
            'reaction_upper':up(delta*E),'arithmetic':'EXACT_RATIONAL_OUTWARD_BINARY64',
            'quantity':'stored_energy_per_out_of_plane_thickness','unit':'J/m',
            'geometry':'CONSTRUCTIVE_ZERO_THICKNESS_STRAIGHT_EDGE_SLIT',
            'representation_error':'ZERO_FOR_DECLARED_RECTANGLE_AND_STRAIGHT_SLIT',
            'physical_model_error':'UNKNOWN','a_exact':str(moments.mesh.a),
            'model_key':f'L4_H1_yc{moments.mesh.yc}',
            'mu_exact':[str(k) for k in mu],'delta_exact':str(delta)}


def release_bounds(left,right,da):
    da=rat(da)
    if da<=0:raise ValueError('positive crack increment required')
    for key in ('model_key','mu_exact','delta_exact'):
        if key not in left or left[key]!=right.get(key):
            raise ValueError('release requires the same geometry family, material and displacement')
    if F(right['a_exact'])-F(left['a_exact'])!=da:
        raise ValueError('release increment must equal the declared nested crack lengths')
    mu=tuple(F(k) for k in left['mu_exact']);delta=F(left['delta_exact'])
    if len(mu)!=3 or min(mu)<=0 or delta<=0:
        raise ValueError('positive scalar material and displacement metadata required')
    # Far-field strip reference, not the exact release of a finite crack.
    compliance=sum((h/k for h,k in zip((F(3,8),F(1,4),F(3,8)),mu)),F(0))
    Gss=delta**2/(2*compliance)
    lo=(rat(left['lower'])-rat(right['upper']))/da
    hi=(rat(left['upper'])-rat(right['lower']))/da
    # This scope assumes the SAME fixed boundary displacement and material,
    # with nested slits. Under those explicit premises Pi decreases.
    if hi<0:raise ArithmeticError('non-nested or inconsistent energy bounds')
    return {'lower':down(max(F(0),lo)),'upper':up(hi),'unit':'J/m^2',
            'quantity':'mean_energy_release_over_finite_step',
            'stationary_Gss_exact':str(Gss),
            'relative_width_at_Gss':float((hi-max(F(0),lo))/Gss)}


def evaluate(nx,a,mu=(1.,1.,1.),*,aggregate=True,independent=False,certify=True,yc=F(0)):
    st=time.perf_counter();m=make_slit(nx,a,yc=yc,aggregate=aggregate);built=time.perf_counter()
    v,psi,cost=solve_fields(m,mu,independent=independent);solved=time.perf_counter()
    moments=exact_moments(m,v,psi) if certify else None
    cert=energy_bounds(moments,mu) if certify else None
    cost.update(mesh_setup_s=built-st,certificate_s=time.perf_counter()-solved,
                full_s=time.perf_counter()-st,triangles=len(m.triangles),nodes=len(m.xy),
                area_exact=str(m.area),aggregation=m.aggregation)
    return m,v,psi,moments,cert,cost
