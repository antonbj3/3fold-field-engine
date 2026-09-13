#include "../../../src/field_engine/edt_pair_quotient_v1/quotient.cuh"
#include <cstdint>
__global__ void dense_floor(int start,int count,int denominator,int* output){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)output[i]=corrected_floor(start+i,denominator);
}
__global__ void paired_floor(const int* numerator,const int* denominator,int count,int* output){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)output[i]=corrected_floor(numerator[i],denominator[i]);
}
extern "C" int quotient_dense(int start,int count,int denominator,int* output) noexcept {
    if(!output||count<1||count>2097153||start<-1048576||static_cast<int64_t>(start)+count-1>1048576||denominator<2||denominator>1022)return 1;
    int* device=nullptr;if(cudaMalloc(&device,size_t(count)*sizeof(int))!=cudaSuccess)return 2;
    dense_floor<<<(count+255)/256,256>>>(start,count,denominator,device);
    int status=cudaGetLastError()!=cudaSuccess?2:0;
    if(!status&&cudaMemcpy(output,device,size_t(count)*sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess)status=2;
    if(cudaFree(device)!=cudaSuccess)status=2;return status;
}
extern "C" int quotient_pairs(const int* numerator,const int* denominator,int count,int* output) noexcept {
    if(!numerator||!denominator||!output||count<1||count>2097153)return 1;
    for(int i=0;i<count;++i)if(numerator[i]<-1048576||numerator[i]>1048576||denominator[i]<2||denominator[i]>1022)return 1;
    int *dn=nullptr,*dd=nullptr,*result=nullptr;int status=0;
    if(cudaMalloc(&dn,size_t(count)*sizeof(int))!=cudaSuccess||cudaMalloc(&dd,size_t(count)*sizeof(int))!=cudaSuccess||cudaMalloc(&result,size_t(count)*sizeof(int))!=cudaSuccess)status=2;
    if(!status&&(cudaMemcpy(dn,numerator,size_t(count)*sizeof(int),cudaMemcpyHostToDevice)!=cudaSuccess||cudaMemcpy(dd,denominator,size_t(count)*sizeof(int),cudaMemcpyHostToDevice)!=cudaSuccess))status=2;
    if(!status){paired_floor<<<(count+255)/256,256>>>(dn,dd,count,result);if(cudaGetLastError()!=cudaSuccess)status=2;}
    if(!status&&cudaMemcpy(output,result,size_t(count)*sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess)status=2;
    if(dn&&cudaFree(dn)!=cudaSuccess)status=2;if(dd&&cudaFree(dd)!=cudaSuccess)status=2;if(result&&cudaFree(result)!=cudaSuccess)status=2;return status;
}
