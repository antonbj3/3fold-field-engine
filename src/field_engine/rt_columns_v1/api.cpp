#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include "params.h"
#include <algorithm>
#include <limits>
#include <cmath>
#include <thread>
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <iterator>
#include <cstdint>

static void check(OptixResult r) {
    if (r != OPTIX_SUCCESS) throw std::runtime_error(optixGetErrorName(r));
}
static void check(cudaError_t r) {
    if (r != cudaSuccess) throw std::runtime_error(cudaGetErrorString(r));
}
static CUdeviceptr allocate(size_t bytes, const void* source = nullptr) {
    void* pointer = nullptr;
    check(cudaMalloc(&pointer, bytes));
    if (source) check(cudaMemcpy(pointer, source, bytes, cudaMemcpyHostToDevice));
    return reinterpret_cast<CUdeviceptr>(pointer);
}
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };

struct State {
    OptixDeviceContext context{};
    cudaStream_t stream{};
    OptixModule module{};
    OptixProgramGroup groups[3]{};
    OptixPipeline pipeline{};
    OptixShaderBindingTable sbt{};
    CUdeviceptr boxes{},queries{},depths{},signs{},vertices{},scratch{},storage{},sbt_data{},ray_data{},output{},params_data{};
    uint32_t capacity{}, hit_capacity{}; double3 anchor{}; double floor_z{}, area_floor{}; WindingParams parameters{};
    std::thread::id owner=std::this_thread::get_id();
    void initialize(const std::string& ptx,const std::vector<double3>& corners) {
        check(cudaFree(nullptr));
        check(optixInit());
        
        OptixDeviceContextOptions options{};
        check(optixDeviceContextCreate(nullptr, &options, &context));
        
        check(cudaStreamCreate(&stream));
        vertices = allocate(corners.size()*sizeof(double3), corners.data());
        std::vector<OptixAabb> bounds(corners.size()/3);
        for(size_t i=0;i<bounds.size();++i){
            double lo[3]={INFINITY,INFINITY,INFINITY},hi[3]={-INFINITY,-INFINITY,-INFINITY};
            for(int j=0;j<3;++j){auto c=corners[3*i+j];double v[3]={c.x-anchor.x,c.y-anchor.y,c.z-anchor.z};
                for(int k=0;k<3;++k){lo[k]=std::min(lo[k],v[k]);hi[k]=std::max(hi[k],v[k]);}}
            float l[3],h[3];
            for(int k=0;k<3;++k){
                float pad=64*std::numeric_limits<float>::epsilon()*std::max({1.,std::abs(lo[k]),std::abs(hi[k])});
                l[k]=std::nextafter(float(lo[k])-pad,-INFINITY);h[k]=std::nextafter(float(hi[k])+pad,INFINITY);}
            bounds[i]={l[0],l[1],l[2],h[0],h[1],h[2]};
        }
        boxes=allocate(bounds.size()*sizeof(OptixAabb),bounds.data());
        const unsigned flags = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;
        OptixBuildInput build{};
        build.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        build.customPrimitiveArray.aabbBuffers = &boxes;
        build.customPrimitiveArray.numPrimitives = bounds.size();
        build.customPrimitiveArray.flags = &flags;
        build.customPrimitiveArray.numSbtRecords = 1;
        OptixAccelBuildOptions accel{};
        accel.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        accel.operation = OPTIX_BUILD_OPERATION_BUILD;
        OptixAccelBufferSizes sizes{};
        check(optixAccelComputeMemoryUsage(context, &accel, &build, 1, &sizes));
        scratch = allocate(sizes.tempSizeInBytes);
        storage = allocate(sizes.outputSizeInBytes);
        OptixTraversableHandle gas{};
        check(optixAccelBuild(context, stream, &accel, &build, 1, scratch,
              sizes.tempSizeInBytes, storage, sizes.outputSizeInBytes, &gas, nullptr, 0));
        OptixPipelineCompileOptions compile{};
        compile.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        compile.numPayloadValues = 1;
        compile.numAttributeValues = 3;
        compile.pipelineLaunchParamsVariableName = "params";
        compile.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
        OptixModuleCompileOptions module_options{};
        
        check(optixModuleCreate(context, &module_options, &compile,
                               ptx.data(), ptx.size(), nullptr, nullptr, &module));
        OptixProgramGroupDesc descriptions[3]{};
        descriptions[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        descriptions[0].raygen = {module, "__raygen__winding"};
        descriptions[1].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        descriptions[1].miss = {module, "__miss__winding"};
        descriptions[2].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        descriptions[2].hitgroup.moduleAH = module;
        descriptions[2].hitgroup.entryFunctionNameAH = "__anyhit__winding";
        
        descriptions[2].hitgroup.moduleIS = module;
        descriptions[2].hitgroup.entryFunctionNameIS = "__intersection__columns";
        OptixProgramGroupOptions group_options{};
        check(optixProgramGroupCreate(context, descriptions, 3, &group_options, nullptr, nullptr, groups));
        OptixPipelineLinkOptions link{};
        link.maxTraceDepth = 1;
        
        check(optixPipelineCreate(context, &compile, &link, groups, 3, nullptr, nullptr, &pipeline));
        OptixStackSizes stack{};
        for (auto group : groups) check(optixUtilAccumulateStackSizes(group, &stack, pipeline));
        unsigned traversal{}, state{}, continuation{};
        check(optixUtilComputeStackSizes(&stack, 1, 0, 0, &traversal, &state, &continuation));
        check(optixPipelineSetStackSize(pipeline, traversal, state, continuation, 1));
        Record records[3]{};
        for (int i=0; i<3; ++i) check(optixSbtRecordPackHeader(groups[i], &records[i]));
        sbt_data = allocate(sizeof(records), records);
        
        sbt.raygenRecord = sbt_data;
        sbt.missRecordBase = sbt_data + sizeof(Record);
        sbt.missRecordStrideInBytes = sizeof(Record);
        sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = sbt_data + 2*sizeof(Record);
        sbt.hitgroupRecordStrideInBytes = sizeof(Record);
        sbt.hitgroupRecordCount = 1;
        ray_data = allocate(size_t(capacity)*sizeof(float3));
        output = allocate(size_t(capacity)*sizeof(int));
        queries=allocate(size_t(capacity)*sizeof(double2));
        depths=allocate(size_t(capacity)*hit_capacity*sizeof(double));
        signs=allocate(size_t(capacity)*hit_capacity*sizeof(int));
        parameters={gas,reinterpret_cast<double3*>(vertices),reinterpret_cast<float3*>(ray_data),
          reinterpret_cast<double2*>(queries),reinterpret_cast<int*>(output),reinterpret_cast<double*>(depths),
          reinterpret_cast<int*>(signs),hit_capacity,area_floor};
        params_data = allocate(sizeof(parameters), &parameters);

    }
    int release() noexcept {
        int failed=0;
        if(stream && cudaStreamSynchronize(stream)!=cudaSuccess) failed=2;
        for(auto ptr:{boxes,queries,depths,signs,params_data,output,ray_data,sbt_data,scratch,storage,vertices})
            if(ptr && cudaFree(reinterpret_cast<void*>(ptr))!=cudaSuccess) failed=2;
        if(pipeline && optixPipelineDestroy(pipeline)!=OPTIX_SUCCESS) failed=2;
        for(auto group:groups) if(group && optixProgramGroupDestroy(group)!=OPTIX_SUCCESS) failed=2;
        if(module && optixModuleDestroy(module)!=OPTIX_SUCCESS) failed=2;
        if(context && optixDeviceContextDestroy(context)!=OPTIX_SUCCESS) failed=2;
        if(stream && cudaStreamDestroy(stream)!=cudaSuccess) failed=2;
        return failed;
    }
};
extern "C" int columns_create(const char* path,const double* input,uint32_t triangles,uint32_t capacity,uint32_t hit_capacity,const double* anchor,void** handle) {
    if(!handle || *handle || !path || !input || !anchor || !triangles || triangles>1000000 || !capacity || capacity>262144 || !hit_capacity || hit_capacity>256) return 1;
    for(size_t i=0;i<size_t(triangles)*9;++i) if(!std::isfinite(input[i]) || std::abs(input[i])>1e12) return 1;
    for(int i=0;i<3;++i) if(!std::isfinite(anchor[i]) || std::abs(anchor[i])>1e12) return 1;
    State* state=nullptr;
    try {
        std::ifstream file(path);std::string ptx{std::istreambuf_iterator<char>(file),{}};if(ptx.empty())return 1;
        std::vector<double3> corners(size_t(triangles)*3);double scale=1.,floor=INFINITY;
        for(size_t i=0;i<corners.size();++i){corners[i]=make_double3(input[3*i],input[3*i+1],input[3*i+2]);
            scale=std::max({scale,std::abs(input[3*i]),std::abs(input[3*i+1])});floor=std::min(floor,input[3*i+2]-anchor[2]);}
        state=new State;state->capacity=capacity;state->hit_capacity=hit_capacity;
        state->anchor=make_double3(anchor[0],anchor[1],anchor[2]);state->floor_z=floor-std::max(1.,std::abs(floor)*1e-4);
        state->area_floor=1e-12*scale*scale;state->initialize(ptx,corners);*handle=state;return 0;
    }catch(...){if(state){state->release();delete state;}return 2;}
}
extern "C" int columns_query(void* handle,const double* xy,uint32_t count,int32_t* counts,double* depths,int32_t* signs) {
    if(!handle || !xy || !counts || !depths || !signs)return 1;
    State& s=*static_cast<State*>(handle);if(s.owner!=std::this_thread::get_id())return 3;
    if(!count || count>s.capacity)return 1;
    for(size_t i=0;i<size_t(count)*2;++i)if(!std::isfinite(xy[i]) || std::abs(xy[i])>1e12)return 1;
    try{
        std::vector<float3> rays(count);for(size_t i=0;i<count;++i)rays[i]=make_float3(xy[2*i]-s.anchor.x,xy[2*i+1]-s.anchor.y,s.floor_z);
        check(cudaMemcpy(reinterpret_cast<void*>(s.ray_data),rays.data(),size_t(count)*sizeof(float3),cudaMemcpyHostToDevice));
        check(cudaMemcpy(reinterpret_cast<void*>(s.queries),xy,size_t(count)*sizeof(double2),cudaMemcpyHostToDevice));
        check(optixLaunch(s.pipeline,s.stream,s.params_data,sizeof(WindingParams),&s.sbt,count,1,1));check(cudaStreamSynchronize(s.stream));
        check(cudaMemcpy(counts,reinterpret_cast<void*>(s.output),size_t(count)*sizeof(int32_t),cudaMemcpyDeviceToHost));
        for(size_t i=0;i<count;++i)if(counts[i]>int(s.hit_capacity))return 4;
        check(cudaMemcpy(depths,reinterpret_cast<void*>(s.depths),size_t(count)*s.hit_capacity*sizeof(double),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(signs,reinterpret_cast<void*>(s.signs),size_t(count)*s.hit_capacity*sizeof(int32_t),cudaMemcpyDeviceToHost));return 0;
    }catch(...){return 2;}
}
extern "C" int columns_destroy(void** handle) {
    if(!handle)return 1;if(!*handle)return 0;State* s=static_cast<State*>(*handle);
    if(s->owner!=std::this_thread::get_id())return 3;int status=s->release();delete s;*handle=nullptr;return status;
}
