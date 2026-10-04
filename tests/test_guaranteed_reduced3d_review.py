"""Independent rational tensor and propagated-uncertainty regression tests."""
from fractions import Fraction as F
from itertools import product
from types import MappingProxyType
import numpy as np
import pytest
from field_engine.experimental import guaranteed_scalar3d as c
from field_engine.experimental import guaranteed_reduced3d as r


def rational_tensor_basis(regions=False):
    A=[[1,F(1,5),0],[0,1,F(-1,7)],[0,0,1]]
    mesh=c.affine_mesh(c.structured_mesh(2),A)
    labels=['a' if sum(mesh.points[k][0] for k in tet)/4<F(3,5) else 'b' for tet in mesh.tets] if regions else None
    basis=r.ReducedCertificate(mesh,F(1,3),degree=2,region_labels=labels)
    v,z,_=c.solve_fields(mesh,F(1,3),1,None,2)
    basis.add(v,z)
    v=list(v);z=list(z)
    _,boundary,_,_=c.potential_layout(mesh,2)
    free=next(i for i in range(len(v)) if i not in boundary)
    v[free]+=F(1,13)
    z[len(z)//2]+=F(1,17)
    z=c.repair_flux(mesh,z,F(1,3))
    basis.add(v,z)
    return basis


def exact_cross_tensor(basis,i,j,d,e,kind,region=0):
    mesh=basis.mesh;dofs=c.potential_layout(mesh,2)[0]
    result=F(0)
    for label,tet,V,row,vdofs in zip(basis.labels,mesh.tets,mesh.volumes,mesh.tet_faces,dofs):
        if label!=basis.regions[region]:continue
        p=[mesh.points[k] for k in tet]
        values=[]
        for field in (basis.fields[i],basis.fields[j]):
            if kind=='G':
                values.append(c.potential_gradients(p,[field[0][k] for k in vdofs],2))
            else:
                flux=[sgn*field[1][k] for k,sgn in row]
                values.append([[sum(flux[k]*(x[a]-p[k][a]) for k in range(4))/(3*V) for a in range(3)] for x in p])
        a,b=values
        result+=V*(sum(a[k][d]*b[k][e] for k in range(4))+sum(x[d] for x in a)*sum(x[e] for x in b))/20
    return result


@pytest.mark.parametrize('kind',['G','Q'])
def test_each_retained_signed_cross_tensor_encloses_its_rational_integral(kind):
    basis=rational_tensor_basis(regions=True);tensor=getattr(basis,'_'+kind)
    nonrepresentable=0
    for region,i,j,d,e in product(range(2),range(2),range(2),range(3),range(3)):
        exact=exact_cross_tensor(basis,i,j,d,e,kind,region)
        mid,rad=tensor[region,i,j,d,e]
        assert F(mid)-F(rad)<=exact<=F(mid)+F(rad)
        nonrepresentable+=exact!=F(mid)
    assert nonrepresentable>0


@pytest.mark.parametrize('kind',['G','Q'])
def test_material_batch_propagates_retained_tensor_uncertainty(kind):
    # Isolate the contraction's interval guarantee with an explicit energy
    # interval fixture. The other tensor remains exact; these are unit tests
    # of uncertainty transport, not admission of serialized certificates.
    mesh=c.structured_mesh(2)
    basis=r.ReducedCertificate(mesh,1,degree=2,diagonal_only=True)
    v,z,_=c.solve_fields(mesh,1,1,None,2);basis.add(v,z)
    G={(0,0,0,d,d):(1.,.25 if kind=='G' else 0.) for d in range(3)}
    Q={(0,0,0,d,d):(2.,.25 if kind=='Q' else 0.) for d in range(3)}
    basis._G=MappingProxyType(G);basis._Q=MappingProxyType(Q)
    cert=basis.bounds_material_batch([{None:F(1,3)}])[0]
    # Three diagonal coefficients: Ehi=(3+3/4)/3, Dhi=(6+3/4)*3.
    expected_lo=1/(F(3,1)+(F(3,4) if kind=='G' else 0))*3
    expected_hi=(F(6,1)+(F(3,4) if kind=='Q' else 0))*3
    assert F(cert['lower'])<=expected_lo
    assert F(cert['upper'])>=expected_hi


@pytest.mark.parametrize('eps',[F(1,2**80),F(1,3**50)])
def test_almost_vertex_simplex_stays_exact_under_sheared_material_edit(eps):
    basis=rational_tensor_basis()
    A=[[F(5,4),F(1,7),0],[0,F(3,4),F(1,9)],[0,0,F(9,8)]]
    alpha=(1-eps,eps);beta=(eps,1-eps)
    k=F(10**12)
    cert=basis.bounds(A,conductivity={None:k},alpha=alpha,beta=beta)
    v=basis.reconstruct(alpha,'potential');z=basis.reconstruct(beta,'flux')
    _,det,_=r.fc._mat(A);m=c.affine_mesh(basis.mesh,A)
    z=tuple(x*det for x in z)
    trial=c.certificate(m,v,z,basis.source,k,None,2,with_energy_moments=True)
    E=sum(F(trial['energy_moments']['primal'][d][d]) for d in range(3))
    L=F(trial['energy_moments']['load_exact'])
    exact=c.certificate(m,tuple(x*L/E for x in v),z,basis.source,k,None,2)
    assert 0<cert['lower']<=exact['lower']<=exact['upper']<=cert['upper']


@pytest.mark.parametrize('family',['series','parallel'])
def test_layered_batch_retains_exact_solution_tightness_for_unequal_weights(family):
    mesh=c.structured_mesh(2,(2,1,1));el=c.box_electrodes(mesh)
    labels=['a' if (sum(mesh.points[i][0] for i in t)/4<1 if family=='series' else sum(mesh.points[i][1] for i in t)/4<F(1,2)) else 'b' for t in mesh.tets]
    basis=r.ReducedCertificate(mesh,0,el,2,labels,diagonal_only=True)
    for anchor in (F(1,10000),F(10000)):
        v,z,_=c.solve_fields(mesh,0,[1 if label=='a' else anchor for label in labels],el,2)
        basis.add(v,z)
    ks=[F(1,7),F(13,7),F(17)]
    answers=basis.bounds_material_batch([{'a':1,'b':k} for k in ks])
    for k,answer in zip(ks,answers):
        true=1+1/k if family=='series' else 4/(1+k)
        assert F(answer['lower'])<=true<=F(answer['upper'])
        assert answer['relative_width_upper']<1e-10
