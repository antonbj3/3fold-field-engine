// Separate larger-contract synchronous exact squared EDT. Host pointers are caller-owned.
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <algorithm>

__global__ void axis_min(const int* source, int* target, int size, int stride, int length) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if(index>=size) return;
    int position=(index/stride)%length, base=index-position*stride, best=2147483647;
    for(int site=0;site<length;++site) {
        int delta=position-site;
        best=min(best,source[base+site*stride]+delta*delta);
    }
    target[index]=best;
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
            axis_min<<<(size+255)/256,256>>>(a,b,size,strides[axis],lengths[axis]);
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
