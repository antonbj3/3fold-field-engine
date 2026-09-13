// Capture immutable RT/mask/EDT work once; replay computes fresh full fields.
#define fused_create base_fused_create
#define fused_query base_fused_query
#define fused_destroy base_fused_destroy
#define fused_pin base_fused_pin
#define fused_unpin base_fused_unpin
#include "../mesh_fused_shared_v1/field.cu"
#undef fused_create
#undef fused_query
#undef fused_destroy
#undef fused_pin
#undef fused_unpin

struct GraphField {
    void* base=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    int* flags=nullptr;bool failed=false;
};
extern "C" int fused_destroy(void** handle) noexcept {
    if(!handle)return 1;if(!*handle)return 0;auto* g=static_cast<GraphField*>(*handle);
    if(g->base){auto& f=*static_cast<Fused*>(g->base);auto& s=*static_cast<State*>(f.rt);
        if(s.owner!=std::this_thread::get_id())return 3;}
    int status=0;
    if(g->executable&&cudaGraphExecDestroy(g->executable)!=cudaSuccess)status=2;
    if(g->graph&&cudaGraphDestroy(g->graph)!=cudaSuccess)status=2;
    if(g->flags&&cudaFreeHost(g->flags)!=cudaSuccess)status=2;
    int prior=base_fused_destroy(&g->base);if(prior)status=prior;
    delete g;*handle=nullptr;return status;
}
extern "C" int fused_create(const char* path,const double* corners,uint32_t triangles,const double* anchor,const double* xy,int nx,int ny,int nz,double pitch,const double* zs,void** handle) noexcept {
    if(!handle||*handle)return 1;GraphField* g=nullptr;bool capturing=false;int phase=0;
    try{
        g=new GraphField;
        phase=10;int status=base_fused_create(path,corners,triangles,anchor,xy,nx,ny,nz,pitch,zs,&g->base);
        if(status){delete g;return status;}
        auto& f=*static_cast<Fused*>(g->base);auto& s=*static_cast<State*>(f.rt);auto& p=*f.pair;
        phase=20;check(cudaMallocHost(&g->flags,sizeof(int)));
        phase=30;check(cudaStreamBeginCapture(s.stream,cudaStreamCaptureModeGlobal));capturing=true;
        check(cudaMemsetAsync(f.flags,0,sizeof(int),s.stream));
        phase=40;check(optixLaunch(s.pipeline,s.stream,s.params_data,sizeof(WindingParams),&s.sbt,f.columns,1,1));
        phase=50;hit_range<<<f.blocks,128,0,s.stream>>>(reinterpret_cast<int*>(s.output),reinterpret_cast<double*>(s.depths),f.columns,s.hit_capacity,f.low,f.high,f.flags);check(cudaGetLastError());
        prepare_range<<<1,1,0,s.stream>>>(f.low,f.high,f.blocks,f.zs,p.nz,f.range,f.fractions);check(cudaGetLastError());
        prepare_keys<<<f.blocks,128,0,s.stream>>>(reinterpret_cast<int*>(s.output),reinterpret_cast<double*>(s.depths),f.columns,s.hit_capacity,f.range,f.keys);check(cudaGetLastError());
        classify_hits<<<(p.size+255)/256,256,0,s.stream>>>(reinterpret_cast<int*>(s.output),reinterpret_cast<int*>(s.signs),f.keys,f.fractions,s.hit_capacity,p.nz,p.size,p.mask,f.flags);check(cudaGetLastError());
        fused_surface<<<(p.size+255)/256,256,0,s.stream>>>(p.mask,f.surface,p.nx,p.ny,p.nz);check(cudaGetLastError());
        int sentinel=(p.nx-1)*(p.nx-1)+(p.ny-1)*(p.ny-1)+(p.nz-1)*(p.nz-1)+1;
        initialise<<<(p.size+255)/256,256,0,s.stream>>>(p.mask,p.a,p.size,sentinel);check(cudaGetLastError());
        int lengths[]={p.nx,p.ny,p.nz},strides[]={p.ny*p.nz,p.nz,1};int* a=p.a;int* b=p.b;
        for(int axis=0;axis<3;++axis){axis_envelope<<<(2*p.size/lengths[axis]+127)/128,128,0,s.stream>>>(a,b,p.size,strides[axis],lengths[axis]);check(cudaGetLastError());std::swap(a,b);}
        signed_output<<<(p.size+255)/256,256,0,s.stream>>>(a,p.distance,p.size,f.pitch);check(cudaGetLastError());

        phase=60;check(cudaMemcpyAsync(g->flags,f.flags,sizeof(int),cudaMemcpyDeviceToHost,s.stream));
        phase=70;auto capture_status=cudaStreamEndCapture(s.stream,&g->graph);capturing=false;check(capture_status);
        phase=80;check(cudaGraphInstantiate(&g->executable,g->graph,0));
        *handle=g;return 0;
    }catch(...){
        try{throw;}catch(const std::exception& error){std::cerr << "Graph preparation phase " << phase << ": " << error.what() << std::endl;}catch(...){}
        if(capturing&&g&&g->base){auto& f=*static_cast<Fused*>(g->base);cudaGraph_t abandoned=nullptr;
            cudaStreamEndCapture(static_cast<State*>(f.rt)->stream,&abandoned);if(abandoned)cudaGraphDestroy(abandoned);}
        if(g){void* h=g;fused_destroy(&h);}return 2;
    }
}
extern "C" int fused_query(void* handle,uint8_t* solid,uint8_t* surface,float* distance) noexcept {
    if(!handle||!solid||!surface||!distance)return 1;
    auto& g=*static_cast<GraphField*>(handle);auto& f=*static_cast<Fused*>(g.base);
    auto& s=*static_cast<State*>(f.rt);auto& p=*f.pair;
    if(s.owner!=std::this_thread::get_id())return 3;if(g.failed)return 2;
    try{
        check(cudaGraphLaunch(g.executable,s.stream));
        check(cudaMemcpyAsync(solid,p.mask,p.size,cudaMemcpyDeviceToHost,s.stream));
        check(cudaMemcpyAsync(surface,f.surface,p.size,cudaMemcpyDeviceToHost,s.stream));
        check(cudaMemcpyAsync(distance,p.distance,p.size*sizeof(float),cudaMemcpyDeviceToHost,s.stream));
        check(cudaStreamSynchronize(s.stream));
        if(*g.flags&12){g.failed=true;return 4;}if((*g.flags&3)!=3){g.failed=true;return 1;}
        return 0;
    }catch(...){cudaStreamSynchronize(s.stream);g.failed=true;return 2;}
}
extern "C" int fused_pin(void* handle,void* region,size_t bytes) noexcept {
    if(!handle)return 1;auto& g=*static_cast<GraphField*>(handle);
    if(g.failed)return 2;return base_fused_pin(g.base,region,bytes);
}
extern "C" int fused_unpin(void* handle,void* region) noexcept {
    if(!handle)return 1;return base_fused_unpin(static_cast<GraphField*>(handle)->base,region);
}
