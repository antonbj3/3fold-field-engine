"""FACIT Poisson/Prandtl cases and monotone subcell boundary regressions."""
import os, sys
from pathlib import Path
import numpy as np
import pytest
from scipy.sparse.linalg import spsolve
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT/'src',ROOT/'src/field_engine',ROOT/'src/field_engine/opt',ROOT/'tests'):
    sys.path.insert(0,str(p))
import laplaceflode_v1 as LF
import cutrand_facit as K


def points(N,dim=2):
    h=1/N
    ax=[(np.arange(int(1.5*N))+.5)*h-.75+.5*h*v for v in (.37,.61,.23)[:dim]]
    return np.stack(np.meshgrid(*ax,indexing='ij'),-1),h


def solve_direct(sd,h,rhs=1,merge=1e-3):
    system=LF.bygg_sdf_dirichlet(sd,h,theta_merge=merge)
    b=h*h*np.asarray(system['P'].T@np.full(system['stats']['nodes'],rhs)).ravel()+system['boundary_rhs']
    x=spsolve(system['A'],b)
    return system['P']@x+system['offset'],system


@pytest.mark.parametrize('shape',['circle','ellipse','square0','square22','square45','triangle'])
def test_facit_torsion_second_order(shape):
    errors=[]
    for N in [32,64,128]:
        P,h=points(N)
        if shape=='circle':
            sd=np.linalg.norm(P,axis=-1)-.5;ex=K.torsion_constant('circle',R=.5)
        elif shape=='ellipse':
            # A smooth implicit function is sufficient for linear zero crossings; no exact SDF claim.
            sd=np.sqrt((P[...,0]/.5)**2+(P[...,1]/.3)**2)-1;ex=K.torsion_constant('ellipse',a=.5,b=.3)
        else:
            if shape=='triangle':
                V=K.equilateral_triangle(1);ex=K.torsion_constant('triangle',side=1)
            else:
                a={'square0':0,'square22':np.pi/8,'square45':np.pi/4}[shape]
                R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
                V=np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]])@R.T
                ex=K.torsion_constant('square',side=1)
            sd=K.polygon_sdf(P,V)
        u,_=solve_direct(sd,h,rhs=2)
        errors.append(abs(2*u.sum()*h*h/ex-1))
    assert errors[-1]<.001
    # Corners have grid-phase fluctuations; lock the quadratic envelope rather than a lucky slope.
    assert all(e < 7/N**2 for e,N in zip(errors,[32,64,128]))


@pytest.mark.parametrize('rotation',[0,.43])
def test_facit_ellipsoid_poisson(rotation):
    a=np.array([.5,.35,.25])
    R=np.array([[np.cos(rotation),-np.sin(rotation),0],[np.sin(rotation),np.cos(rotation),0],[0,0,1]])
    P,h=points(24,3);Q=P@R
    sd=np.sqrt(np.sum((Q/a)**2,axis=-1))-1
    field,system,info=LF.los_sdf_poisson(sd,h)
    assert info==0
    assert abs(np.nansum(field)*h**3/K.ellipsoid_poisson_integral(a)-1)<.012
    assert np.nanmin(field)>0
    assert system['stats']['relative_residual']<1e-10


@pytest.mark.parametrize('epsilon',[1e-5,1e-12,1e-200])
def test_small_theta_merge_preserves_m_matrix_and_solution(epsilon):
    sd=np.array([1.,-epsilon,-(1+epsilon),-(2+epsilon),1.])
    u,s=solve_direct(sd,1.)
    A=s['A'].toarray()
    assert s['stats']['merged_nodes']==1
    assert np.allclose(A,A.T,rtol=0,atol=1e-14)
    assert np.all((A-np.diag(np.diag(A)))<=0)
    assert np.linalg.eigvalsh(A).min()>0
    assert u.min()>=0
    assert u[0]==pytest.approx(epsilon/(1+2*epsilon)*u[1],rel=1e-12,abs=1e-250)
    field,_,info=LF.los_sdf_poisson(sd,1,rhs=0,randvarde=3)
    assert info==0 and np.allclose(field[sd<0],3,atol=2e-12)


def test_unmergeable_sliver_is_refused():
    with pytest.raises(ValueError,match='sliver'):
        LF.bygg_sdf_dirichlet(np.array([1.,-1e-8,1.]),1.)


@pytest.mark.parametrize('sd,h',[([1,-1,np.nan],1),([1,-1,1],0),([-1,-1,1],1),([1,1,1],1)])
def test_bad_geometry_is_refused(sd,h):
    with pytest.raises(ValueError): LF.bygg_sdf_dirichlet(np.array(sd),h)


def test_failed_cg_does_not_publish_field():
    P,h=points(32)
    sd=np.linalg.norm(P,axis=-1)-.5
    u,_,info=LF.los_sdf_poisson(sd,h,maxiter=1)
    assert info!=0 and np.isnan(u[sd<0]).all()


def test_cpu_mesh_sdf_retains_real_subcell_plane_position():
    import trimesh,warp as wp
    import faltkarna_v1_mesh_to_sdf as M
    wp.config.enable_cuda=False; wp.init()
    mesh=trimesh.creation.box([2,2,2]);mesh.apply_translation([.13,.17,.19])
    lo=np.array([-2.,-2.,-2.]);h=.25
    gmin,shape,_,_,sd=M.surface_raster_and_flood(mesh.vertices,mesh.faces,h,lo,metod='winding_cpu',wp=wp)
    origin=lo+gmin*h
    ijk=np.array([np.argmin(abs(origin[k]+np.arange(shape[k])*h-v)) for k,v in enumerate([.13,.17,.19])])
    for x in [1.,1.25]:
        ijk[0]=round((x-origin[0])/h)
        # Existing winding kernel jitters the query by <1e-4*h; keep that documented bound.
        assert sd[tuple(ijk)]==pytest.approx(x-1.13,abs=1e-4*h)


def test_1d_green_function_is_piecewise_linear_through_tiny_cut():
    # -u''=delta(x-c), u(a)=u(b)=0. The continuous Green function is affine on each side of c.
    a,b,c=1.-1e-6,4.4,3.
    x=np.arange(6,dtype=float)
    sd=np.maximum(a-x,x-b)
    source=np.zeros_like(x);source[int(c)]=1.
    field,system,info=LF.los_sdf_poisson(sd,1.,rhs=source)
    inside=sd<0;xx=x[inside]
    expected=np.minimum(xx-a,c-a)*(b-np.maximum(xx,c))/(b-a)
    assert info==0 and system['stats']['merged_nodes']==1
    assert np.allclose(field[inside],expected,rtol=1e-10,atol=1e-14)


# --- review additions --------------------------------------------------------------------------
def _box_sd(P, lo, hi):
    q = np.maximum(lo - P, P - hi)
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(np.max(q, -1), 0)


def _square_series(nmax=401):
    i = np.arange(1, nmax, 2.0); I, J = np.meshgrid(i, i, indexing='ij')
    return float(np.sum(64 / (np.pi**6 * I**2 * J**2 * (I**2 + J**2))))


@pytest.mark.parametrize('eps', [1e-9, 2.36e-5, 2e-4])
def test_grid_aligned_box_corner_is_solved_not_refused(eps):
    """Faces on lattice planes, domain enlarged by eps*h (float noise, or the winding query's
    1e-4*pitch jitter): every merge candidate of a corner node is itself subthreshold. Before the
    review change this raised 'sliver' for any eps > 0; the unmerged answer equals theta_merge=0."""
    N = 16; h = 1 / N; ax = np.arange(-2, N + 3) * h
    P = np.stack(np.meshgrid(ax, ax, indexing='ij'), -1)
    sd = _box_sd(P, -eps * h, 1 + eps * h)
    u, s, info = LF.los_sdf_poisson(sd, h)
    u0, _, info0 = LF.los_sdf_poisson(sd, h, theta_merge=0.0)
    assert info == 0 and info0 == 0 and s['stats']['unmerged_small_nodes'] > 0
    assert np.nanmin(u) > 0
    exact = _square_series()
    assert abs(np.nansum(u) * h * h / exact - 1) < 0.02
    assert np.nansum(u) == pytest.approx(np.nansum(u0), rel=1e-4)   # merged edge nodes differ O(theta)


def test_independent_square_torsion_series_matches_fixture():
    """FACIT's Saint-Venant square against an independent double sine series of -lap u = 1."""
    assert 4 * _square_series() == pytest.approx(K.torsion_constant('square', side=1), rel=1e-7)
