"""Complete gate parity against trimesh, adverse fixtures and native refusal controls."""
import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/field_engine'))
sys.path.insert(0,str(ROOT/'tests'))
import faltkarna_v1_mesh_to_sdf as M
import mesh_gate_native_v1 as G
from mesh_gate_cases import box,fixtures,random_cases
CXX=shutil.which(os.environ.get('CXX','c++'))
FLAGS=['-std=c++17','-O3','-fopenmp','-fno-fast-math','-ffp-contract=off','-shared','-fPIC']

@pytest.fixture(scope='session')
def libraries(tmp_path_factory):
    if CXX is None:pytest.skip('no C++ compiler')
    folder=tmp_path_factory.mktemp('gate_libs');libs={}
    for name,src in [('FIELD_MESH_GATE_LIBRARY','mesh_gate_native_v1/gate.cpp'),('FIELD_MESH_SDF_LIBRARY','mesh_sdf_native_v1/field.cpp')]:
        out=folder/(name+'.so')
        subprocess.run([CXX,*FLAGS,str(ROOT/'src/field_engine'/src),'-o',str(out)],check=True,capture_output=True,timeout=120)
        libs[name]=str(out)
    return libs

@pytest.fixture
def native(libraries,monkeypatch):
    for key,value in libraries.items():monkeypatch.setenv(key,value)
    return libraries


def equal(ref,got):
    for key,val in ref.items():
        if key=='avstand_till_yta_median_mm':
            assert abs(val-got[key]) <= 1e-4,(key,val,got[key])
        else:assert val==got.get(key),(key,val,got.get(key))

@pytest.mark.parametrize('name,V,T,pitch',list(fixtures()),ids=lambda x:x if isinstance(x,str) else None)
def test_adverse_decisions_and_field_bits(native,name,V,T,pitch):
    lo=V.min(0)-3*pitch
    ref=M.surface_raster_and_flood(V,T,pitch,lo,metod='raypar_vindning')
    got=M.surface_raster_and_flood(V,T,pitch,lo,metod='raypar_vindning_native')
    for a,b in zip(ref,got):np.testing.assert_array_equal(a,b)
    gm,shape,y,solid,sd=ref
    kw=dict(lo=lo,gmin=gm,sd=sd)
    r=M.vattentathetsgrind(V,T,solid,pitch,**kw)
    if 'djupmatt_fel' in r:
        with pytest.warns(RuntimeWarning):g=M.vattentathetsgrind(V,T,solid,pitch,backend='native',**kw)
    else:
        g=M.vattentathetsgrind(V,T,solid,pitch,backend='native',**kw)
        assert g.get('grind_backend')=='native'
        assert 'grind_fallback' not in g
    equal(r,g)


def test_random_mutations(native):
    for name,V,T,pitch in random_cases():
        lo=V.min(0)-3*pitch
        gm,_,_,solid,sd=M.surface_raster_and_flood(V,T,pitch,lo,metod='raypar_vindning_native')
        kw=dict(lo=lo,gmin=gm,sd=sd)
        r=M.vattentathetsgrind(V,T,solid,pitch,**kw)
        if 'djupmatt_fel' in r:
            with pytest.warns(RuntimeWarning):g=M.vattentathetsgrind(V,T,solid,pitch,backend='native',**kw)
        else:g=M.vattentathetsgrind(V,T,solid,pitch,backend='native',**kw)
        equal(r,g)

@pytest.mark.parametrize('count',[0,1,2,15,16,95,96,101])
def test_same_probe_population_and_rng(count):
    rng=np.random.default_rng(2718)
    for shape in [(2,3,4),(9,11,13)]:
        for solid in [rng.random(shape)<.5,np.ones(shape,bool),np.zeros(shape,bool)]:
            a=M._grind_probe_indices(solid,count,20260902)
            b=M._grind_probe_indices(solid,count,20260902,compact=True)
            np.testing.assert_array_equal(a,b)


def test_volume_bytes(native):
    for name,V,T,pitch in fixtures():
        m,*_=G.facts(V,T)
        assert np.float64(G.volume(m)).tobytes()==np.float64(m.volume).tobytes(),name


def test_missing_library_fallback(native,monkeypatch):
    V,T=box();solid=np.zeros((9,9,9),bool);solid[3:6,3:6,3:6]=True
    monkeypatch.delenv('FIELD_MESH_GATE_LIBRARY')
    r=M.vattentathetsgrind(V,T,solid,1.)
    with pytest.warns(RuntimeWarning,match='trimesh used instead'):
        g=M.vattentathetsgrind(V,T,solid,1.,backend='native')
    equal(r,g);assert g['grind_backend']=='trimesh'

@pytest.mark.parametrize('kind',['flat_v','flat_t','negative','oob','float_faces','empty'])
def test_unsupported_gate_falls_back(native,kind):
    V,T=box();solid=np.zeros((3,3,3),bool)
    if kind=='flat_v':V=V.ravel()
    if kind=='flat_t':T=T.ravel()
    if kind=='negative':T[0,0]=-1
    if kind=='oob':T[0,0]=10000
    if kind=='float_faces':T=T.astype(float)
    if kind=='empty':T=T[:0]
    r=M.vattentathetsgrind(V,T,solid,1.)
    with pytest.warns(RuntimeWarning):g=M.vattentathetsgrind(V,T,solid,1.,backend='native')
    equal(r,g)


def test_flat_vertices_field_falls_back_safely(native):
    V,T=box();V=V.ravel();T=T+8
    with pytest.raises(IndexError):M.surface_raster_and_flood(V,T,.5,np.zeros(3),metod='raypar_vindning')
    with pytest.warns(RuntimeWarning),pytest.raises(IndexError):
        M.surface_raster_and_flood(V,T,.5,np.zeros(3),metod='raypar_vindning_native')
    import mesh_sdf_native_v1 as N
    with pytest.raises(N.NativeUnavailable):N.field_stage(V,T,.5,np.zeros(3))


def test_library_replacement_refused(native,tmp_path,monkeypatch):
    file=tmp_path/'lib.so';shutil.copyfile(native['FIELD_MESH_GATE_LIBRARY'],file)
    monkeypatch.setenv('FIELD_MESH_GATE_LIBRARY',str(file));G.lib()
    with file.open('ab') as f:f.write(b'changed')
    with pytest.raises(RuntimeError,match='Restart'):G.lib()


def test_independent_parity_near_thresholds(native):
    V,T=box((0.,0.,0.),(4.,4.,4.))
    rng=np.random.default_rng(43);P=rng.normal(size=(500,3))*4
    extra=[]
    for z in [0.,4.,4.-1e-7,np.nextafter(4.-1e-7,0.),np.nextafter(4.-1e-7,9.)]:
        for x in [0.,2.,4.,-4e-7,np.nextafter(-4e-7,-1.)]:extra.append([x,2.,z])
    P=np.vstack((P,extra))
    np.testing.assert_array_equal(M.ray_parity_probe(V,T,P),G.parity(V,T,P))


def test_depth_threshold_reference_order(native):
    import trimesh
    pytest.importorskip('rtree')
    V,T=box((0.,0.,0.),(4.,4.,4.));m=trimesh.Trimesh(V,T,process=False);m.merge_vertices()
    P=np.array([[x,2,2] for x in [1.,np.nextafter(1.,0.),np.nextafter(1.,2.),-1.,np.nextafter(-1.,0.)]])
    ref=trimesh.proximity.closest_point(m,P)[1];got=G.distances(m,P,1.)
    np.testing.assert_array_equal(ref>=1.,got>=1.)


def test_full_op_modes(native):
    V,T=box();lo=V.min(0)-1.5
    for faces in [T,T[:-2]]:
        r=M.mesh_to_sdf_del(None,V,faces,.5,0.,lo,'cpu',metod='raypar_vindning_native',grind_backend='trimesh',returnera_falt=True)
        g=M.mesh_to_sdf_del(None,V,faces,.5,0.,lo,'cpu',metod='raypar_vindning_native',returnera_falt=True)
        equal({k:v for k,v in r['vattentathet'].items() if k!='grind_wall_s'},g['vattentathet'])
        np.testing.assert_array_equal(r['solid_final'],g['solid_final'])
        np.testing.assert_array_equal(r['sd'].view(np.uint32),g['sd'].view(np.uint32))
        if r['vattentathet']['status']!='OK':
            with pytest.raises(ValueError,match=r['vattentathet']['status']):
                M.mesh_to_sdf_del(None,V,faces,.5,0.,lo,'cpu',metod='raypar_vindning_native',grind='strikt')
    g=M.mesh_to_sdf_del(None,V,T,.5,0.,lo,'cpu',metod='raypar_vindning_native',grind='av')
    assert g['vattentathet'] is None


def test_sanitizer(tmp_path):
    if CXX is None:pytest.skip('no C++ compiler')
    exe=tmp_path/'gate_sanitizer'
    flags=['-std=c++17','-O1','-g','-fopenmp','-fno-fast-math','-ffp-contract=off',
           '-fno-omit-frame-pointer','-fsanitize=address,undefined','-fno-sanitize-recover=all','-no-pie']
    subprocess.run([CXX,*flags,str(ROOT/'tests/native_mesh_gate_sanitizer.cpp'),'-o',str(exe)],check=True,capture_output=True,timeout=120)
    subprocess.run([str(exe)],check=True,capture_output=True,timeout=30)


def test_parity_determinant_rounding_margin(native):
    # det at the exact legacy1e-9 cutoff, and a strongly cancellation-conditioned det.
    rng=np.random.default_rng(8901)
    triangles=[np.array([[0.,0.,0.],[1.,0.,0.],[0.,h,0.]])
               for h in [np.nextafter(1e-9,0.),1e-9,np.nextafter(1e-9,1.)]]
    triangles.append(np.array([[0.,0.,0.],[1e5,1e5,0.],[1e5,np.nextafter(1e5,1e6),0.]]))
    for V in triangles:
        T=np.array([[0,1,2]],np.int64)
        P=np.vstack((rng.normal(size=(100,3)), V.mean(0)+[0,0,-1.]))
        np.testing.assert_array_equal(M.ray_parity_probe(V,T,P),G.parity(V,T,P))


def test_native_aabb_candidates_equal_rtree(native):
    import trimesh
    pytest.importorskip('rtree')
    m=trimesh.creation.icosphere(subdivisions=2,radius=3.)
    m.apply_transform(trimesh.transformations.rotation_matrix(.371,[1,2,3]))
    rng=np.random.default_rng(518);points=rng.uniform(-4,4,(80,3));width=rng.uniform(.001,3,(80,1))
    tree=G._Tree(m)
    try:ids,counts=tree.intersection_v(points-width,points+width)
    finally:tree.close()
    native_sets=np.array_split(ids,np.cumsum(counts)[:-1])
    for i,bounds in enumerate(np.column_stack((points-width,points+width))):
        assert set(native_sets[i])==set(m.triangles_tree.intersection(bounds))


def test_auto_gate_without_rtree_selects_trimesh(native,monkeypatch):
    import importlib.util
    import warnings
    real=importlib.util.find_spec
    monkeypatch.setattr(importlib.util,'find_spec',lambda name,*a:None if name=='rtree' else real(name,*a))
    assert M._auto_grind_backend({'backend':'native_cpu'})=='trimesh'
    V,T=box();lo=V.min(0)-1.5
    with warnings.catch_warnings():
        warnings.simplefilter('error',RuntimeWarning)
        g=M.mesh_to_sdf_del(None,V,T,.5,0.,lo,'cpu',metod='raypar_vindning_native',returnera_falt=True)
    assert 'grind_backend' not in g['vattentathet'] and 'grind_fallback' not in g['vattentathet']


def test_auto_gate_selection(native):
    expect='native' if __import__('importlib').util.find_spec('rtree') else 'trimesh'
    assert M._auto_grind_backend({'backend':'native_cpu'})==expect
    assert M._auto_grind_backend({'backend':'numpy'})=='trimesh'


def test_tree_query_rejects_malformed_boxes(native):
    import trimesh
    m=trimesh.creation.icosphere(subdivisions=1);tree=G._Tree(m)
    try:
        with pytest.raises(RuntimeError,match=r'\(n, 6\)'):tree.intersection_v(np.zeros((3,2)),np.ones((3,2)))
    finally:tree.close()
