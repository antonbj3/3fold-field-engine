// Reuse frozen setup and tracing; add a separate compact readback symbol.
#include "../rt_columns_v1/api.cpp"
extern "C" int columns_query_compact(void* handle,const double* xy,uint32_t count,int32_t* counts,double* depths,int32_t* signs,uint32_t* width) {
    if(!handle || !xy || !counts || !depths || !signs || !width)return 1;
    State& s=*static_cast<State*>(handle);if(s.owner!=std::this_thread::get_id())return 3;
    if(!count || count>s.capacity)return 1;
    for(size_t i=0;i<size_t(count)*2;++i)if(!std::isfinite(xy[i]) || std::abs(xy[i])>1e12)return 1;
    try{
        std::vector<float3> rays(count);for(size_t i=0;i<count;++i)rays[i]=make_float3(xy[2*i]-s.anchor.x,xy[2*i+1]-s.anchor.y,s.floor_z);
        check(cudaMemcpy(reinterpret_cast<void*>(s.ray_data),rays.data(),size_t(count)*sizeof(float3),cudaMemcpyHostToDevice));
        check(cudaMemcpy(reinterpret_cast<void*>(s.queries),xy,size_t(count)*sizeof(double2),cudaMemcpyHostToDevice));
        check(optixLaunch(s.pipeline,s.stream,s.params_data,sizeof(WindingParams),&s.sbt,count,1,1));check(cudaStreamSynchronize(s.stream));
        check(cudaMemcpy(counts,reinterpret_cast<void*>(s.output),size_t(count)*sizeof(int32_t),cudaMemcpyDeviceToHost));
        uint32_t maximum=0;
        for(size_t i=0;i<count;++i){if(counts[i]>int(s.hit_capacity))return 4;maximum=std::max(maximum,uint32_t(counts[i]));}
        if(maximum){
            check(cudaMemcpy2D(depths,maximum*sizeof(double),reinterpret_cast<void*>(s.depths),s.hit_capacity*sizeof(double),maximum*sizeof(double),count,cudaMemcpyDeviceToHost));
            check(cudaMemcpy2D(signs,maximum*sizeof(int32_t),reinterpret_cast<void*>(s.signs),s.hit_capacity*sizeof(int32_t),maximum*sizeof(int32_t),count,cudaMemcpyDeviceToHost));
        }
        *width=maximum;return 0;
    }catch(...){return 2;}
}
