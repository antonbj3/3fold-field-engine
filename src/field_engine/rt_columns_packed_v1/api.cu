// Preserve frozen RT setup/tracing; pack strided hits on device before readback.
#define columns_create reference_columns_create
#define columns_query reference_columns_query
#define columns_destroy reference_columns_destroy
#include "../rt_columns_v1/api.cpp"
#undef columns_create
#undef columns_query
#undef columns_destroy
struct PackedState {void* reference=nullptr;double* depths=nullptr;int32_t* signs=nullptr;};
__global__ void pack_rows(const int32_t* counts,const double* source_z,const int32_t* source_s,
                          double* z,int32_t* signs,uint32_t count,uint32_t width,uint32_t capacity) {
    uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count*width)return;
    uint32_t row=i/width,column=i%width;
    if(column<static_cast<uint32_t>(counts[row])){z[i]=source_z[row*capacity+column];signs[i]=source_s[row*capacity+column];}
    else {z[i]=0.;signs[i]=0;}
}
extern "C" int columns_destroy(void** handle) {
    if(!handle)return 1;if(!*handle)return 0;
    auto* p=static_cast<PackedState*>(*handle);auto* s=static_cast<State*>(p->reference);
    if(s->owner!=std::this_thread::get_id())return 3;
    int status=0;
    if(p->depths && cudaFree(p->depths)!=cudaSuccess)status=2;
    if(p->signs && cudaFree(p->signs)!=cudaSuccess)status=2;
    int prior=reference_columns_destroy(&p->reference);if(prior)status=prior;
    delete p;*handle=nullptr;return status;
}
extern "C" int columns_create(const char* path,const double* input,uint32_t triangles,uint32_t capacity,uint32_t hit_capacity,const double* anchor,void** handle) {
    if(!handle||*handle)return 1;
    PackedState* p=nullptr;
    try {
        p=new PackedState;
        int status=reference_columns_create(path,input,triangles,capacity,hit_capacity,anchor,&p->reference);
        if(status){delete p;return status;}
        check(cudaMalloc(&p->depths,size_t(capacity)*hit_capacity*sizeof(double)));
        check(cudaMalloc(&p->signs,size_t(capacity)*hit_capacity*sizeof(int32_t)));
        *handle=p;return 0;
    }catch(...){if(p){if(p->reference){void* h=p;columns_destroy(&h);}else delete p;}return 2;}
}
extern "C" int columns_query(void* handle,const double* xy,uint32_t count,int32_t* counts,double* depths,int32_t* signs) {
    if(!handle)return 1;
    return reference_columns_query(static_cast<PackedState*>(handle)->reference,xy,count,counts,depths,signs);
}
extern "C" int columns_query_compact(void* handle,const double* xy,uint32_t count,int32_t* counts,double* depths,int32_t* signs,uint32_t* width) {
    if(!handle||!xy||!counts||!depths||!signs||!width)return 1;
    auto& p=*static_cast<PackedState*>(handle);auto& s=*static_cast<State*>(p.reference);
    if(s.owner!=std::this_thread::get_id())return 3;
    if(!count||count>s.capacity)return 1;
    for(size_t i=0;i<size_t(count)*2;++i)if(!std::isfinite(xy[i])||std::abs(xy[i])>1e12)return 1;
    try {
        std::vector<float3> rays(count);
        for(size_t i=0;i<count;++i)rays[i]=make_float3(xy[2*i]-s.anchor.x,xy[2*i+1]-s.anchor.y,s.floor_z);
        check(cudaMemcpy(reinterpret_cast<void*>(s.ray_data),rays.data(),size_t(count)*sizeof(float3),cudaMemcpyHostToDevice));
        check(cudaMemcpy(reinterpret_cast<void*>(s.queries),xy,size_t(count)*sizeof(double2),cudaMemcpyHostToDevice));
        check(optixLaunch(s.pipeline,s.stream,s.params_data,sizeof(WindingParams),&s.sbt,count,1,1));
        check(cudaStreamSynchronize(s.stream));
        check(cudaMemcpy(counts,reinterpret_cast<void*>(s.output),size_t(count)*sizeof(int32_t),cudaMemcpyDeviceToHost));
        uint32_t maximum=0;
        for(size_t i=0;i<count;++i){if(counts[i]<0||counts[i]>int(s.hit_capacity))return 4;maximum=std::max(maximum,uint32_t(counts[i]));}
        if(maximum){
            pack_rows<<<(count*maximum+255)/256,256>>>(reinterpret_cast<int32_t*>(s.output),reinterpret_cast<double*>(s.depths),reinterpret_cast<int32_t*>(s.signs),p.depths,p.signs,count,maximum,s.hit_capacity);
            check(cudaGetLastError());
            check(cudaMemcpy(depths,p.depths,size_t(count)*maximum*sizeof(double),cudaMemcpyDeviceToHost));
            check(cudaMemcpy(signs,p.signs,size_t(count)*maximum*sizeof(int32_t),cudaMemcpyDeviceToHost));
        }
        *width=maximum;return 0;
    }catch(...){return 2;}
}
