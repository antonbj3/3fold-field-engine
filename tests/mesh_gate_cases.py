"""Independent adversarial corpus for the complete gate, public/analytic only."""
import numpy as np
import trimesh
F = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
              [2,3,7],[2,7,6],[1,2,6],[1,6,5],[3,0,4],[3,4,7]],np.int64)

def box(lo=(-3.,-3.,-3.),hi=(3.,3.,3.)):
    x,y,z=lo;X,Y,Z=hi
    v=np.array([[x,y,z],[X,y,z],[X,Y,z],[x,Y,z],[x,y,Z],[X,y,Z],[X,Y,Z],[x,Y,Z]],np.float64)
    return v,F.copy()

def join(*parts):
    v,f,off=[],[],0
    for a,b in parts:v.append(a);f.append(b+off);off+=len(a)
    return np.vstack(v),np.vstack(f)

def fixtures():
    v,f=box()
    yield 'closed',v,f,.5
    yield 'hole',v,f[:-2],.5
    yield 'duplicate_face',v,np.vstack((f,f[:1])),.5
    yield 'duplicate_shell',*join((v,f),(v,f)),.5
    yield 'inverted',v,f[:,::-1],.5
    ff=f.copy();ff[2:4]=ff[2:4,::-1]
    yield 'inverted_patch',v,ff,.5
    yield 'unmerged',v[f].reshape(-1,3),np.arange(f.size).reshape(-1,3),.5
    yield 'self_intersection',*join(box(),box((-1,-1,-1),(5,5,5))),.5
    yield 'nested_shell',*join(box(),(v*.5,f[:,::-1])),.5
    yield 'vertex_nonmanifold',*join(box(),box((3,3,3),(6,6,6))),.5
    # Subdivide an edge in just one face: true geometric boundary has a T junction.
    vv=np.vstack((v,(v[0]+v[2])/2));ff=np.vstack((f[1:],[[0,8,1],[8,2,1]]))
    yield 't_junction',vv,ff,.5
    yield 'repeated_vertex_face',v,np.vstack((f,[[0,0,1],[2,2,2]])),.5
    vv=np.vstack((v,[[0,0,0],[1,0,0],[2,1e-14,0]]))
    yield 'sliver',vv,np.vstack((f,[[8,9,10]])),.5
    for delta in [1e-9,4e-9,6e-9,1.1e-8]:
        vv=v[f].reshape(-1,3);vv[0,0]+=delta
        yield f'weld_bin_{delta:g}',vv,np.arange(f.size).reshape(-1,3),.5
    for shift in [0.,np.nextafter(0.,1.),np.nextafter(.5,0.)-.5,np.nextafter(.5,1.)-.5]:
        vv,ff=box((0,0,0),(6,6,6));vv[ff[-1],2]+=shift
        yield f'open_sample_plane_{shift:g}',vv,ff[:-1],.5
    yield 'offset_1e6',v+1e6,f,.5
    for angle in [.27,.91]:
        rot=trimesh.transformations.rotation_matrix(angle,[1,2,3])[:3,:3]
        yield f'rotated_{angle:g}',v@rot.T,f,.37
    m=trimesh.creation.icosphere(subdivisions=1,radius=3.)
    vv=np.asarray(m.vertices);ff=np.asarray(m.faces)
    yield 'sphere',vv,ff,.37


def random_cases(n=100):
    rng=np.random.default_rng(20261001)
    for i in range(n):
        m=trimesh.creation.icosphere(subdivisions=1,radius=3.)
        v=np.asarray(m.vertices).copy();f=np.asarray(m.faces).copy()
        v=v*(.8+rng.random(3)*.4)+rng.normal(0,.01,v.shape)
        op=i%8
        if op==1:f=np.delete(f,rng.choice(len(f),size=1+i%5,replace=False),axis=0)
        if op==2:f=np.vstack((f,f[rng.choice(len(f),size=1+i%4)]))
        if op==3:f=f[:,::-1]
        if op==4:
            rows=rng.choice(len(f),size=1+i%7,replace=False);f[rows]=f[rows,::-1]
        if op==5:v,f=join((v,f),(v*.7+.5,f[:,::-1]))
        if op==6:v=v[f].reshape(-1,3);f=np.arange(len(v)).reshape(-1,3)
        if op==7:
            f=np.vstack((f,[[0,0,1],[2,2,2]]))
        yield f'random_{i:03}',v,f,.7
