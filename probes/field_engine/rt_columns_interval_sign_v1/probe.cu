#include "../../../src/field_engine/rt_columns_interval_sign_v1/filter.cuh"
#include <cstdint>
#include <initializer_list>
__global__ void evaluate_intervals(const double3* triangles,const double2* queries,int count,float* lower,float* upper,uint8_t* rejected,double* edges){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    double3 a=triangles[3*i],b=triangles[3*i+1],c=triangles[3*i+2];double2 q=queries[i];
    Interval result[3];interval_numerators(a,b,c,q,result);
    for(int j=0;j<3;++j){lower[3*i+j]=result[j].lo;upper[3*i+j]=result[j].hi;}
    rejected[i]=interval_outside(a,b,c,q);
    edges[3*i]=(b.x-q.x)*(c.y-q.y)-(b.y-q.y)*(c.x-q.x);
    edges[3*i+1]=(c.x-q.x)*(a.y-q.y)-(c.y-q.y)*(a.x-q.x);
    edges[3*i+2]=(a.x-q.x)*(b.y-q.y)-(a.y-q.y)*(b.x-q.x);
}
extern "C" int interval_probe(const double* triangles,const double* queries,int count,float* lower,float* upper,uint8_t* rejected,double* edges) noexcept {
    if(!triangles||!queries||!lower||!upper||!rejected||!edges||count<1||count>1048576)return 1;
    double *dt=nullptr,*dq=nullptr,*de=nullptr;float *dl=nullptr,*du=nullptr;uint8_t* dr=nullptr;int status=0;
    #define CHECK(x) if((x)!=cudaSuccess){status=2;goto cleanup;}
    CHECK(cudaMalloc(&dt,size_t(count)*9*sizeof(double)));CHECK(cudaMalloc(&dq,size_t(count)*2*sizeof(double)));
    CHECK(cudaMalloc(&de,size_t(count)*3*sizeof(double)));CHECK(cudaMalloc(&dl,size_t(count)*3*sizeof(float)));
    CHECK(cudaMalloc(&du,size_t(count)*3*sizeof(float)));CHECK(cudaMalloc(&dr,count));
    CHECK(cudaMemcpy(dt,triangles,size_t(count)*9*sizeof(double),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dq,queries,size_t(count)*2*sizeof(double),cudaMemcpyHostToDevice));
    evaluate_intervals<<<(count+255)/256,256>>>(reinterpret_cast<double3*>(dt),reinterpret_cast<double2*>(dq),count,dl,du,dr,de);CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(lower,dl,size_t(count)*3*sizeof(float),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(upper,du,size_t(count)*3*sizeof(float),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(rejected,dr,count,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(edges,de,size_t(count)*3*sizeof(double),cudaMemcpyDeviceToHost));
    cleanup:
    for(void* p:{static_cast<void*>(dt),static_cast<void*>(dq),static_cast<void*>(de),static_cast<void*>(dl),static_cast<void*>(du),static_cast<void*>(dr)})if(p&&cudaFree(p)!=cudaSuccess)status=2;
    return status;
    #undef CHECK
}
