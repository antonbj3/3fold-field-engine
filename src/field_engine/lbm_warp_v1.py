"""Resident float64 D3Q19 GPU execution with ordered arithmetic and host sampling.

Kernel compilation is a separate one-time build cost. Each run prepares and uploads
its own geometry/forcing, advances two resident kernels per step, samples the same
convergence statistic and returns the complete host distribution.
"""
import operator
import hashlib
from pathlib import Path
import numpy as np
import warp as wp
import lbm_domare_v1 as reference

wp.set_module_options({'enable_backward': False, 'fast_math': False, 'fuse_fp': False})
D = wp.float64
_LOADED_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def implementation_binding(device='cuda:0'):
    """Bind calibration reuse to the loaded source, compiler options and CUDA runtime."""
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != _LOADED_SOURCE_SHA256:
        raise RuntimeError('Reload the GPU LBM module after a source change')
    options = dict(wp.get_module_options(__name__))
    if any(options.get(k) is not False for k in ('fast_math', 'fuse_fp', 'enable_backward')):
        raise RuntimeError('GPU LBM requires the validated ordered compiler options')
    target = wp.get_device(device)
    if not target.is_cuda:
        raise ValueError('A CUDA device is required')
    return dict(source_sha256=_LOADED_SOURCE_SHA256, warp=wp.config.version,
                device=target.name, architecture=target.arch,
                cuda_toolkit=list(wp.get_cuda_toolkit_version()),
                cuda_driver=list(wp.get_cuda_driver_version()), module_options=options)


@wp.func
def _density(f: wp.array(dtype=D), n: int, i: int):
    r0=f[i]+f[8*n+i];r1=f[n+i]+f[9*n+i]
    r2=f[2*n+i]+f[10*n+i];r3=f[3*n+i]+f[11*n+i]
    r4=f[4*n+i]+f[12*n+i];r5=f[5*n+i]+f[13*n+i]
    r6=f[6*n+i]+f[14*n+i];r7=f[7*n+i]+f[15*n+i]
    rho=((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))
    rho=rho+f[16*n+i];rho=rho+f[17*n+i];rho=rho+f[18*n+i]
    result=D(1)
    if rho>D(1e-9):result=rho
    return result


@wp.func
def _moment(f: wp.array(dtype=D), e: wp.array(dtype=wp.int32), n: int, i: int):
    x=D(0);y=D(0);z=D(0)
    for q in range(19):
        value=f[q*n+i]
        x=x+value*D(e[3*q]);y=y+value*D(e[3*q+1]);z=z+value*D(e[3*q+2])
    return wp.vec3d(x,y,z)


@wp.kernel
def _collide(f: wp.array(dtype=D), solid: wp.array(dtype=wp.int32),
             force: wp.array(dtype=D), e: wp.array(dtype=wp.int32),
             ea: wp.array(dtype=D), weights: wp.array(dtype=D), pw: wp.array(dtype=D),
             n: int, tau: D, u: wp.array(dtype=D), fstar: wp.array(dtype=D)):
    i=wp.tid();rho=_density(f,n,i);moment=_moment(f,e,n,i)
    ax=force[i];ay=force[n+i];az=force[2*n+i]
    x=(moment[0]+(D(.5)*rho)*ax)/rho
    y=(moment[1]+(D(.5)*rho)*ay)/rho
    z=(moment[2]+(D(.5)*rho)*az)/rho
    if solid[i]!=0:x=D(0);y=D(0);z=D(0)
    u[i]=x;u[n+i]=y;u[2*n+i]=z
    usq=((D(0)+x*x)+y*y)+z*z
    ua=((D(0)+x*ax)+y*ay)+z*az
    for q in range(19):
        c=((D(0)+x*D(e[3*q]))+y*D(e[3*q+1]))+z*D(e[3*q+2])
        index=q*n+i
        equilibrium=(weights[q]*rho)*(((D(1)+D(3)*c)+D(4.5)*(c*c))-D(1.5)*usq)
        forcing=(pw[q]*rho)*(D(3)*(ea[index]-ua)+(D(9)*c)*ea[index])
        fstar[index]=(f[index]-((f[index]-equilibrium)/tau))+forcing


@wp.kernel
def _stream_boundaries(fstar: wp.array(dtype=D), indices: wp.array(dtype=wp.int64),
                      boundary_source: wp.array(dtype=wp.int32), boundary_rho: wp.array(dtype=D),
                      u: wp.array(dtype=D), e: wp.array(dtype=wp.int32),
                      weights: wp.array(dtype=D), n: int, output: wp.array(dtype=D)):
    i=wp.tid();source=boundary_source[i]
    if source>=0:
        x=u[source];y=u[n+source];z=u[2*n+source]
        usq=((D(0)+x*x)+y*y)+z*z
        rho=boundary_rho[i]
        for q in range(19):
            c=((D(0)+x*D(e[3*q]))+y*D(e[3*q+1]))+z*D(e[3*q+2])
            output[q*n+i]=(weights[q]*rho)*(((D(1)+D(3)*c)+D(4.5)*(c*c))-D(1.5)*usq)
    else:
        for q in range(19):output[q*n+i]=fstar[indices[q*n+i]]


@wp.kernel
def _maximum_speed(f: wp.array(dtype=D), e: wp.array(dtype=wp.int32), n: int,
                   maximum: wp.array(dtype=D), invalid: wp.array(dtype=wp.int32)):
    i=wp.tid();rho=_density(f,n,i);moment=_moment(f,e,n,i)
    value=wp.abs(moment[0]/rho)
    if value!=value:wp.atomic_max(invalid,0,1)
    else:wp.atomic_max(maximum,0,value)


class PreparedRun:
    def __init__(self, shape, solid, tau, force, boundaries, *, initial=None, device='cuda:0'):
        self.shape=tuple(operator.index(v) for v in shape)
        if len(self.shape)!=3 or any(v<=0 for v in self.shape):
            raise ValueError('Three positive grid dimensions required')
        self.n=int(np.prod(self.shape))
        if self.n*19>np.iinfo(np.int32).max:
            raise ValueError('GPU indexing requires fewer than 2**31 populations')
        self.tau=float(tau)
        if not np.isfinite(self.tau) or self.tau==0:
            raise ValueError('Finite nonzero relaxation time required')
        mask=np.array(solid,dtype=bool,order='C',copy=True)
        forcing=np.array(force,dtype=np.float64,order='C',copy=True)
        if mask.shape!=self.shape or forcing.shape!=self.shape+(3,) or not np.isfinite(forcing).all():
            raise ValueError('Finite forcing and matching solid geometry required')
        if initial is None:
            initial=reference.feq3d(np.ones(self.shape),np.zeros(self.shape+(3,)))
        if (not isinstance(initial,np.ndarray) or initial.dtype!=np.float64 or
            initial.shape!=self.shape+(19,) or not np.isfinite(initial).all()):
            raise ValueError('Finite float64 initial distribution matching the grid required')
        cells=np.arange(self.n,dtype=np.int64).reshape(self.shape)
        indices=np.empty(self.shape+(19,),dtype=np.int64)
        for q in range(19):indices[...,q]=np.roll(cells,reference.Ei[q],axis=(0,1,2))*19+q
        indices[mask]=cells[mask][:,None]*19+reference.OPP
        indices=indices.reshape(self.n,19).T.copy()
        indices=(indices//19+(indices%19)*self.n).ravel()
        sources=np.full(self.shape,-1,dtype=np.int32)
        densities=np.zeros(self.shape)
        for boundary in boundaries:
            density=float(boundary['rho'])
            if not np.isfinite(density):raise ValueError('Finite boundary density required')
            sources[boundary['idx']]=cells[boundary['adj']]
            densities[boundary['idx']]=density
        implementation_binding(device)
        wp.init();self.device=wp.get_device(device)
        if not self.device.is_cuda:raise ValueError('A CUDA device is required')
        def upload(a,dtype):return wp.array(np.ascontiguousarray(a).ravel(),dtype=dtype,device=self.device)
        self.e=upload(reference.E.astype(np.int32),wp.int32)
        self.weights=upload(reference.W,D)
        pref=1-1/(2*self.tau)
        self.pw=upload(pref*reference.W,D)
        self.force=upload(forcing.reshape(self.n,3).T,D)
        self.ea=upload((forcing@reference.E.T).reshape(self.n,19).T,D)
        self.solid=upload(mask.astype(np.int32),wp.int32)
        self.indices=upload(indices,wp.int64)
        self.boundary_source=upload(sources,wp.int32)
        self.boundary_rho=upload(densities,D)
        self.f=upload(initial.reshape(self.n,19).T,D)
        self.fstar=wp.empty(self.n*19,dtype=D,device=self.device)
        self.next=wp.empty(self.n*19,dtype=D,device=self.device)
        self.u=wp.empty(self.n*3,dtype=D,device=self.device)
        self.maximum=wp.zeros(1,dtype=D,device=self.device)
        self.invalid=wp.zeros(1,dtype=wp.int32,device=self.device)

    def step(self):
        wp.launch(_collide,dim=self.n,inputs=[self.f,self.solid,self.force,self.e,self.ea,
            self.weights,self.pw,self.n,self.tau,self.u,self.fstar],device=self.device)
        wp.launch(_stream_boundaries,dim=self.n,inputs=[self.fstar,self.indices,self.boundary_source,
            self.boundary_rho,self.u,self.e,self.weights,self.n,self.next],device=self.device)
        self.f,self.next=self.next,self.f

    def max_speed(self):
        self.maximum.zero_();self.invalid.zero_()
        wp.launch(_maximum_speed,dim=self.n,inputs=[self.f,self.e,self.n,self.maximum,self.invalid],device=self.device)
        value=float(self.maximum.numpy()[0])
        return float('nan') if self.invalid.numpy()[0] else value

    def state(self):
        return self.f.numpy().reshape((19,)+self.shape).transpose(1,2,3,0).copy()


def run_lbm(shp, solid, tau, a_field, bc_planes, steps, sample_every=200, tol=1e-7, *, device='cuda:0'):
    context=PreparedRun(shp,solid,tau,a_field,bc_planes,device=device)
    last=0.0;s=0
    for s in range(steps):
        context.step()
        if s%sample_every==0 and s>0:
            um=context.max_speed()
            if abs(um-last)<tol*max(um,1e-30):break
            last=um
    return context.state(),s
