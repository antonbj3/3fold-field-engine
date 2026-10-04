"""Regression tests for the standalone experimental continuum certificate."""
from fractions import Fraction as F
import dataclasses
from decimal import Decimal
import math
import numpy as np
import pytest
from field_engine.experimental.guaranteed_scalar import (
    Mesh,poisson_certificate,resistance_certificate,scalar_readout,
    solve_poisson_fields,lift_grid_potential,scalar_error_upper,down,up,rat)


def square():
    xy=[(i/2,j/2) for j in range(3) for i in range(3)]
    ts=[]
    for j in range(2):
        for i in range(2):
            a=j*3+i;ts.extend([(a,a+1,a+4),(a,a+4,a+3)])
    return Mesh.admit(xy,ts)


def test_poisson_constructor_no_facit_input():
    m=square();v,p=solve_poisson_fields(m);c=poisson_certificate(m,v,p)
    # Classic square Prandtl J, validation-only reference.
    assert c['lower']<=.140577014956<=c['upper']
    assert abs(c['gap_upper']-(c['upper']-c['lower']))<1e-15
    assert c['representation_error']=='UNKNOWN'


def test_arbitrary_inexact_fields_preserve_upper_and_lower():
    m=square();rng=np.random.default_rng(84)
    for _ in range(10):
        v=rng.normal(size=9);v[list(m.boundary)]=0;p=rng.normal(size=9)
        c=poisson_certificate(m,v,p)
        assert c['lower']<=.140577014956<=c['upper']
        assert abs(c['gap_upper']-(c['upper']-c['lower']))<3e-14


def test_exact_sheet_resistance_and_trace_rejection():
    m=Mesh.admit([(0,0),(2,0),(2,1),(0,1)],[(0,1,2),(0,2,3)])
    arcs={'bottom':[0,1],'right':[1,2],'top':[2,3],'left':[3,0]}
    v=[0,1,1,0];p=[0,0,1,1];c=resistance_certificate(m,v,p,arcs)
    assert c['lower']==c['upper']==2
    p[2]=math.nextafter(1,0)
    with pytest.raises(ValueError):resistance_certificate(m,v,p,arcs)


@pytest.mark.parametrize('k',[0.,-1.,math.nan,math.inf])
def test_invalid_coefficients(k):
    with pytest.raises(ValueError):poisson_certificate(square(),np.zeros(9),np.zeros(9),k=k)


def test_invalid_trace_topology_orientation_and_unadmitted_mesh():
    m=square();v=np.zeros(9);v[0]=1e-100
    with pytest.raises(ValueError):poisson_certificate(m,v,np.zeros(9))
    with pytest.raises(ValueError):Mesh.admit(m.xy,[t[::-1] for t in m.triangles])
    with pytest.raises(ValueError):Mesh.admit(m.xy,list(m.triangles)+[m.triangles[0]])
    with pytest.raises(ValueError):poisson_certificate(Mesh(m.xy,m.triangles,m.boundary),np.zeros(9),np.zeros(9))


def test_exact_outward_rounding_and_scaled_interval():
    for x in [F(1,3),F(-1,3),F(1,2**1080),F(-1,2**1080)]:
        assert F.from_float(down(x))<=x<=F.from_float(up(x))
    c=scalar_readout({'lower':.1,'upper':.10001},3)
    assert c['absolute_error_upper']>=max(abs(c['estimate']-c['lower']),abs(c['upper']-c['estimate']))


def test_native_lift_requires_successful_interior_values():
    m=square();axis=np.array([0.,.5,1.]);field=np.zeros((3,3));field[1,1]=.2
    v=lift_grid_potential(m,[axis,axis],field,np.ones((3,3),bool));_,p=solve_poisson_fields(m)
    c=poisson_certificate(m,v,p);assert c['lower']<=.140577014956<=c['upper']
    field[1,1]=np.nan
    with pytest.raises(ValueError):lift_grid_potential(m,[axis,axis],field,np.ones((3,3),bool))


def test_raw_native_readout_error_is_outward():
    cert={'lower':.1,'upper':.10001};raw=.099
    b=scalar_error_upper(cert,raw,3.)
    assert F.from_float(b)>=max(abs(F.from_float(raw)-3*F.from_float(cert['lower'])),abs(3*F.from_float(cert['upper'])-F.from_float(raw)))


def test_replaced_mesh_is_not_admitted():
    # dataclasses.replace copied the admission token; a doubled triangle list then
    # gave a "certified" lower bound 0.267 above the true J=0.1406 (review probe).
    m=square();v,p=solve_poisson_fields(m)
    for bad in (dataclasses.replace(m,triangles=m.triangles+m.triangles),
                dataclasses.replace(m,boundary=m.boundary[:1]),dataclasses.replace(m)):
        with pytest.raises(ValueError):poisson_certificate(bad,v,p)
        with pytest.raises(ValueError):solve_poisson_fields(bad)


def test_inputs_are_exact_or_rejected():
    assert rat(2**53+1)==2**53+1 and rat(np.int64(7))==7 and rat(Decimal('0.5'))==F(1,2)
    assert rat(np.float32(0.1))==F.from_float(float(np.float32(0.1)))
    for x in (Decimal('0.1'),np.longdouble(1)/np.longdouble(3),'2',True):
        with pytest.raises((ValueError,TypeError)):rat(x)
    m=square();v,p=solve_poisson_fields(m)
    assert poisson_certificate(m,v,p,k=np.int64(1))['upper']==poisson_certificate(m,v,p)['upper']
    with pytest.raises((ValueError,TypeError)):poisson_certificate(m,v,p,source='2')


def test_readout_requires_order_and_keeps_scope():
    with pytest.raises(ValueError):scalar_readout({'lower':5.,'upper':1.})
    m=square();v,p=solve_poisson_fields(m);r=scalar_readout(poisson_certificate(m,v,p))
    assert r['geometry']=='EXACT_REPRESENTED_POLYGON' and r['representation_error']=='UNKNOWN'
