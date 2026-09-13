// Device-resident RT hits -> exact composite-key mask -> paired signed EDT.
#include "../rt_columns_v1/api.cpp"
#include "../edt_persistent_pair_v1/edt.cu"

__global__ void hit_range(const int* counts,const double* depths,int columns,int capacity,double* low,double* high,int* flags){
    __shared__ double lows[128],highs[128];
    int col=blockIdx.x*blockDim.x+threadIdx.x;double lo=INFINITY,hi=-INFINITY;
    if(col<columns){int n=counts[col];if(n<0||n>capacity)atomicOr(flags,4);n=max(0,min(n,capacity));
        for(int j=0;j<n;++j){double z=depths[col*capacity+j];if(!isfinite(z))atomicOr(flags,8);else{lo=fmin(lo,z);hi=fmax(hi,z);}}}
    lows[threadIdx.x]=lo;highs[threadIdx.x]=hi;__syncthreads();
    for(int step=64;step;step/=2){if(threadIdx.x<step){lows[threadIdx.x]=fmin(lows[threadIdx.x],lows[threadIdx.x+step]);highs[threadIdx.x]=fmax(highs[threadIdx.x],highs[threadIdx.x+step]);}__syncthreads();}
    if(threadIdx.x==0){low[blockIdx.x]=lows[0];high[blockIdx.x]=highs[0];}
}
__global__ void prepare_range(const double* low,const double* high,int blocks,const double* zs,int nz,double* range,double* fractions){
    double lo=INFINITY,hi=-INFINITY;for(int i=0;i<blocks;++i){lo=fmin(lo,low[i]);hi=fmax(hi,high[i]);}
    double span=1.;if(!isfinite(lo))lo=0.;else span=fmax(__dsub_rn(hi,lo),1e-12);
    range[0]=lo;range[1]=span;
    for(int z=0;z<nz;++z){double f=__dadd_rn(.25,__ddiv_rn(__dmul_rn(.5,__dsub_rn(zs[z],lo)),span));fractions[z]=fmax(0.,fmin(.999,f));}
}
__global__ void prepare_keys(const int* counts,const double* depths,int columns,int capacity,const double* range,double* keys){
    int col=blockIdx.x*blockDim.x+threadIdx.x;if(col>=columns)return;
    int n=max(0,min(counts[col],capacity));double base=__dadd_rn(static_cast<double>(col),.25);
    for(int j=0;j<n;++j)keys[col*capacity+j]=__dadd_rn(base,__ddiv_rn(__dmul_rn(.5,__dsub_rn(depths[col*capacity+j],range[0])),range[1]));
}
__global__ void classify_hits(const int* counts,const int* signs,const double* keys,const double* fractions,int capacity,int nz,int size,uint8_t* mask,int* flags){
    int i=blockIdx.x*blockDim.x+threadIdx.x;bool inside=false,valid=i<size;
    if(valid){int col=i/nz,n=max(0,min(counts[col],capacity)),sum=0;double q=__dadd_rn(static_cast<double>(col),fractions[i%nz]);
        for(int j=0;j<n;++j)if(keys[col*capacity+j]>q)sum+=signs[col*capacity+j];inside=sum!=0;mask[i]=inside;}
    unsigned yes=__ballot_sync(0xffffffff,valid&&inside),no=__ballot_sync(0xffffffff,valid&&!inside);
    if((threadIdx.x&31)==0)atomicOr(flags,(yes?2:0)|(no?1:0));
}
__global__ void fused_surface(const uint8_t* mask,uint8_t* surface,int nx,int ny,int nz){
    int i=blockIdx.x*blockDim.x+threadIdx.x,size=nx*ny*nz;if(i>=size)return;
    int z=i%nz,y=(i/nz)%ny,x=i/(ny*nz);
    surface[i]=mask[i]&&(x==0||x==nx-1||y==0||y==ny-1||z==0||z==nz-1||
        !mask[i-ny*nz]||!mask[i+ny*nz]||!mask[i-nz]||!mask[i+nz]||!mask[i-1]||!mask[i+1]);
}
struct Fused {
    void* rt=nullptr;Pair* pair=nullptr;double *low=nullptr,*high=nullptr,*range=nullptr,*keys=nullptr,*fractions=nullptr,*zs=nullptr;
    uint8_t* surface=nullptr;int* flags=nullptr;double pitch=0;int columns=0,blocks=0;bool failed=false;
};
extern "C" int fused_destroy(void** handle) noexcept {
    if(!handle)return 1;if(!*handle)return 0;auto* f=static_cast<Fused*>(*handle);
    if(f->rt && static_cast<State*>(f->rt)->owner!=std::this_thread::get_id())return 3;
    int status=0;
    if(f->pair){for(void* p:{static_cast<void*>(f->pair->mask),static_cast<void*>(f->pair->a),static_cast<void*>(f->pair->b),static_cast<void*>(f->pair->distance)})if(p&&cudaFree(p)!=cudaSuccess)status=2;delete f->pair;}
    for(void* p:{static_cast<void*>(f->low),static_cast<void*>(f->high),static_cast<void*>(f->range),static_cast<void*>(f->keys),static_cast<void*>(f->fractions),static_cast<void*>(f->zs),static_cast<void*>(f->surface),static_cast<void*>(f->flags)})if(p&&cudaFree(p)!=cudaSuccess)status=2;
    int prior=columns_destroy(&f->rt);if(prior)status=prior;delete f;*handle=nullptr;return status;
}
extern "C" int fused_create(const char* path,const double* corners,uint32_t triangles,const double* anchor,const double* xy,int nx,int ny,int nz,double pitch,const double* zs,void** handle) noexcept {
    if(!handle||*handle||!xy||!zs||nx<1||ny<1||nz<1||nx>512||ny>512||nz>512||static_cast<int64_t>(nx)*ny*nz>1048576||nx*ny>262144||!std::isfinite(pitch)||pitch<1e-12||pitch>1e12)return 1;
    for(int i=0;i<2*nx*ny;++i)if(!std::isfinite(xy[i])||std::abs(xy[i])>1e12)return 1;
    for(int i=0;i<nz;++i)if(!std::isfinite(zs[i]))return 1;
    Fused* f=nullptr;
    try{
        f=new Fused;f->pitch=pitch;f->columns=nx*ny;f->blocks=(f->columns+127)/128;
        int status=columns_create(path,corners,triangles,f->columns,64,anchor,&f->rt);
        if(status){delete f;return status;}
        auto& s=*static_cast<State*>(f->rt);
        f->pair=static_cast<Pair*>(edt_pair_create(nx,ny,nz));if(!f->pair)throw 2;
        check(cudaMalloc(&f->low,f->blocks*sizeof(double)));check(cudaMalloc(&f->high,f->blocks*sizeof(double)));check(cudaMalloc(&f->range,2*sizeof(double)));
        check(cudaMalloc(&f->keys,size_t(f->columns)*s.hit_capacity*sizeof(double)));check(cudaMalloc(&f->fractions,nz*sizeof(double)));check(cudaMalloc(&f->zs,nz*sizeof(double)));
        check(cudaMalloc(&f->surface,f->pair->size));check(cudaMalloc(&f->flags,sizeof(int)));
        std::vector<float3> rays(f->columns);for(int i=0;i<f->columns;++i)rays[i]=make_float3(xy[2*i]-s.anchor.x,xy[2*i+1]-s.anchor.y,s.floor_z);
        check(cudaMemcpy(reinterpret_cast<void*>(s.ray_data),rays.data(),rays.size()*sizeof(float3),cudaMemcpyHostToDevice));
        check(cudaMemcpy(reinterpret_cast<void*>(s.queries),xy,size_t(f->columns)*sizeof(double2),cudaMemcpyHostToDevice));
        check(cudaMemcpy(f->zs,zs,nz*sizeof(double),cudaMemcpyHostToDevice));*handle=f;return 0;
    }catch(...){if(f){void* h=f;fused_destroy(&h);}return 2;}
}
extern "C" int fused_query(void* handle,uint8_t* solid,uint8_t* surface,float* distance) noexcept {
    if(!handle||!solid||!surface||!distance)return 1;auto& f=*static_cast<Fused*>(handle);auto& s=*static_cast<State*>(f.rt);auto& p=*f.pair;
    if(s.owner!=std::this_thread::get_id())return 3;if(f.failed)return 2;
    try{
        check(cudaMemset(f.flags,0,sizeof(int)));
        check(optixLaunch(s.pipeline,s.stream,s.params_data,sizeof(WindingParams),&s.sbt,f.columns,1,1));check(cudaStreamSynchronize(s.stream));
        hit_range<<<f.blocks,128>>>(reinterpret_cast<int*>(s.output),reinterpret_cast<double*>(s.depths),f.columns,s.hit_capacity,f.low,f.high,f.flags);check(cudaGetLastError());
        prepare_range<<<1,1>>>(f.low,f.high,f.blocks,f.zs,p.nz,f.range,f.fractions);check(cudaGetLastError());
        prepare_keys<<<f.blocks,128>>>(reinterpret_cast<int*>(s.output),reinterpret_cast<double*>(s.depths),f.columns,s.hit_capacity,f.range,f.keys);check(cudaGetLastError());
        classify_hits<<<(p.size+255)/256,256>>>(reinterpret_cast<int*>(s.output),reinterpret_cast<int*>(s.signs),f.keys,f.fractions,s.hit_capacity,p.nz,p.size,p.mask,f.flags);check(cudaGetLastError());
        fused_surface<<<(p.size+255)/256,256>>>(p.mask,f.surface,p.nx,p.ny,p.nz);check(cudaGetLastError());
        int sentinel=(p.nx-1)*(p.nx-1)+(p.ny-1)*(p.ny-1)+(p.nz-1)*(p.nz-1)+1;
        initialise<<<(p.size+255)/256,256>>>(p.mask,p.a,p.size,sentinel);check(cudaGetLastError());
        int lengths[]={p.nx,p.ny,p.nz},strides[]={p.ny*p.nz,p.nz,1};int* a=p.a;int* b=p.b;
        for(int axis=0;axis<3;++axis){axis_envelope<<<(2*p.size/lengths[axis]+127)/128,128>>>(a,b,p.size,strides[axis],lengths[axis]);check(cudaGetLastError());std::swap(a,b);}
        signed_output<<<(p.size+255)/256,256>>>(a,p.distance,p.size,f.pitch);check(cudaGetLastError());
        int flags=0;check(cudaMemcpy(&flags,f.flags,sizeof(int),cudaMemcpyDeviceToHost));
        if(flags&12){f.failed=true;return 4;}if((flags&3)!=3){f.failed=true;return 1;}
        check(cudaMemcpy(solid,p.mask,p.size,cudaMemcpyDeviceToHost));check(cudaMemcpy(surface,f.surface,p.size,cudaMemcpyDeviceToHost));check(cudaMemcpy(distance,p.distance,p.size*sizeof(float),cudaMemcpyDeviceToHost));
        return 0;
    }catch(...){f.failed=true;return 2;}
}
