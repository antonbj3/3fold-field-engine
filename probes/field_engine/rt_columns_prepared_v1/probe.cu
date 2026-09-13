#include "../../../src/field_engine/rt_columns_prepared_v1/filter.cuh"
#include "../../../src/field_engine/rt_columns_prepared_v1/host_bounds.hpp"
#include <cstdint>
#include <initializer_list>
__global__ void evaluate_prepared(const double3* triangles,const double2* queries,const float4* geometry,const float4* query,int count,float* lower,float* upper,uint8_t* rejected,double* edges){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    Interval result[3];interval_numerators(geometry[3*i],geometry[3*i+1],geometry[3*i+2],query[i],result);
    for(int j=0;j<3;++j){lower[3*i+j]=result[j].lo;upper[3*i+j]=result[j].hi;}
    rejected[i]=interval_outside(geometry[3*i],geometry[3*i+1],geometry[3*i+2],query[i]);
    double3 a=triangles[3*i],b=triangles[3*i+1],c=triangles[3*i+2];double2 q=queries[i];
    edges[3*i]=(b.x-q.x)*(c.y-q.y)-(b.y-q.y)*(c.x-q.x);
    edges[3*i+1]=(c.x-q.x)*(a.y-q.y)-(c.y-q.y)*(a.x-q.x);
    edges[3*i+2]=(a.x-q.x)*(b.y-q.y)-(a.y-q.y)*(b.x-q.x);
}
extern "C" int prepared_probe(const double* triangles,const double* queries,const double* anchors,int count,float* geometry,float* query,float* lower,float* upper,uint8_t* rejected,double* edges) noexcept {
    if(!triangles||!queries||!anchors||!geometry||!query||!lower||!upper||!rejected||!edges||count<1||count>1048576)return 1;
    for(int i=0;i<count*9;++i)if(!std::isfinite(triangles[i])||std::abs(triangles[i])>1e12)return 1;
    for(int i=0;i<count*3;++i)if(!std::isfinite(anchors[i])||std::abs(anchors[i])>1e12)return 1;
    for(int i=0;i<count*2;++i)if(!std::isfinite(queries[i])||std::abs(queries[i])>1e12)return 1;
    for(int i=0;i<count;++i){double3 anchor=make_double3(anchors[3*i],anchors[3*i+1],anchors[3*i+2]);
        for(int j=0;j<3;++j)reinterpret_cast<float4*>(geometry)[3*i+j]=host_interval_point(triangles[9*i+3*j],triangles[9*i+3*j+1],anchor);
        reinterpret_cast<float4*>(query)[i]=host_interval_point(queries[2*i],queries[2*i+1],anchor);}
    double *dt=nullptr,*dq=nullptr,*de=nullptr;float *dg=nullptr,*dp=nullptr,*dl=nullptr,*du=nullptr;uint8_t* dr=nullptr;int status=0;
    #define CHECK(x) if((x)!=cudaSuccess){status=2;goto cleanup;}
    CHECK(cudaMalloc(&dt,size_t(count)*9*sizeof(double)));CHECK(cudaMalloc(&dq,size_t(count)*2*sizeof(double)));
    CHECK(cudaMalloc(&de,size_t(count)*3*sizeof(double)));CHECK(cudaMalloc(&dg,size_t(count)*12*sizeof(float)));CHECK(cudaMalloc(&dp,size_t(count)*4*sizeof(float)));
    CHECK(cudaMalloc(&dl,size_t(count)*3*sizeof(float)));CHECK(cudaMalloc(&du,size_t(count)*3*sizeof(float)));CHECK(cudaMalloc(&dr,count));
    CHECK(cudaMemcpy(dt,triangles,size_t(count)*9*sizeof(double),cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dq,queries,size_t(count)*2*sizeof(double),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dg,geometry,size_t(count)*12*sizeof(float),cudaMemcpyHostToDevice));CHECK(cudaMemcpy(dp,query,size_t(count)*4*sizeof(float),cudaMemcpyHostToDevice));
    evaluate_prepared<<<(count+255)/256,256>>>(reinterpret_cast<double3*>(dt),reinterpret_cast<double2*>(dq),reinterpret_cast<float4*>(dg),reinterpret_cast<float4*>(dp),count,dl,du,dr,de);CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(lower,dl,size_t(count)*3*sizeof(float),cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(upper,du,size_t(count)*3*sizeof(float),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(rejected,dr,count,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(edges,de,size_t(count)*3*sizeof(double),cudaMemcpyDeviceToHost));
    cleanup:
    for(void* p:{static_cast<void*>(dt),static_cast<void*>(dq),static_cast<void*>(de),static_cast<void*>(dg),static_cast<void*>(dp),static_cast<void*>(dl),static_cast<void*>(du),static_cast<void*>(dr)})if(p&&cudaFree(p)!=cudaSuccess)status=2;
    return status;
    #undef CHECK
}
