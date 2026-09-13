// Separate exact integer-envelope EDT. Host contract and allocation scope unchanged.
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <algorithm>

// First integer position where the new parabola is strictly smaller.
__device__ int separation(int q,int v,int fq,int fv) {
    int numerator=fq+q*q-fv-v*v,denominator=2*(q-v);
    int floor=numerator>=0?numerator/denominator:-((-numerator+denominator-1)/denominator);
    return floor+1;
}
__global__ void axis_envelope(const int* source,int* target,int size,int stride,int length) {
    int line=blockIdx.x*blockDim.x+threadIdx.x;
    if(line>=size/length)return;
    int base=(line/stride)*length*stride+line%stride;
    int sites[512],starts[513];int k=0;
    sites[0]=0;starts[0]=-2147483647;starts[1]=2147483647;
    for(int q=1;q<length;++q) {
        int start=separation(q,sites[k],source[base+q*stride],source[base+sites[k]*stride]);
        while(start<=starts[k]) {--k;start=separation(q,sites[k],source[base+q*stride],source[base+sites[k]*stride]);}
        ++k;sites[k]=q;starts[k]=start;starts[k+1]=2147483647;
    }
    k=0;
    for(int q=0;q<length;++q) {
        while(starts[k+1]<=q)++k;
        int delta=q-sites[k];target[base+q*stride]=source[base+sites[k]*stride]+delta*delta;
    }
}

// Status: 0 success, 1 invalid input, 2 runtime/allocation failure.
// No retained buffers/context ownership or implicit output files.
extern "C" int edt_squared_large(const uint8_t* mask,int nx,int ny,int nz,int32_t* output) {
    if(!mask || !output || nx<1 || ny<1 || nz<1 || nx>512 || ny>512 || nz>512) return 1;
    int size=nx*ny*nz;
    if(size>1048576) return 1;
    int ones=0;
    for(int i=0;i<size;++i) { if(mask[i]>1) return 1; ones+=mask[i]; }
    if(ones==0 || ones==size) return 1;
    int *a=nullptr,*b=nullptr;
    int status=2;
    try {
        int sentinel=(nx-1)*(nx-1)+(ny-1)*(ny-1)+(nz-1)*(nz-1)+1;
        std::vector<int> initial(size);
        for(int i=0;i<size;++i) initial[i]=mask[i] ? sentinel : 0;
        if(cudaMalloc(&a,size*sizeof(int))!=cudaSuccess) throw 2;
        if(cudaMalloc(&b,size*sizeof(int))!=cudaSuccess) throw 2;
        if(cudaMemcpy(a,initial.data(),size*sizeof(int),cudaMemcpyHostToDevice)!=cudaSuccess) throw 2;
        int lengths[]={nx,ny,nz},strides[]={ny*nz,nz,1};
        for(int axis=0;axis<3;++axis) {
            axis_envelope<<<(size/lengths[axis]+127)/128,128>>>(a,b,size,strides[axis],lengths[axis]);
            if(cudaGetLastError()!=cudaSuccess) throw 2;
            std::swap(a,b);
        }
        if(cudaMemcpy(output,a,size*sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess) throw 2;
        status=0;
    } catch(...) { status=2; }
    if(a && cudaFree(a)!=cudaSuccess) status=2;
    if(b && cudaFree(b)!=cudaSuccess) status=2;
    return status;
}
