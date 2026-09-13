// Persistent paired EDT with explicit ownership and correctly rounded signed output.
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <algorithm>
// First integer position where the new parabola is strictly smaller.
__device__ int separation(int q,int v,int fq,int fv) {
    int numerator=fq+q*q-fv-v*v,denominator=2*(q-v);
    int floor=numerator>=0?numerator/denominator:-((-numerator+denominator-1)/denominator);
    return floor+1;
}
__global__ void axis_envelope(const int* source,int* target,int size,int stride,int length) {
    int line=blockIdx.x*blockDim.x+threadIdx.x;
    int lines=size/length; if(line>=2*lines)return;
    int polarity=line/lines;line%=lines;
    int base=polarity*size+(line/stride)*length*stride+line%stride;
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


__global__ void initialise(const uint8_t* mask,int* output,int size,int sentinel) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=size)return;
    output[i]=mask[i]?0:sentinel;output[size+i]=mask[i]?sentinel:0;
}
__global__ void signed_output(const int* pair,float* output,int size,double pitch) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=size)return;
    float outside=__double2float_rn(__dmul_rn(__dsqrt_rn(static_cast<double>(pair[i])),pitch));
    float inside=__double2float_rn(__dmul_rn(__dsqrt_rn(static_cast<double>(pair[size+i])),pitch));
    float delta=__fsub_rn(outside,inside),sign=delta>0?1.f:(delta<0?-1.f:0.f);
    output[i]=__fsub_rn(delta,__fmul_rn(sign,__double2float_rn(__dmul_rn(.5,pitch))));
}
struct Pair {int nx,ny,nz,size;uint8_t* mask=nullptr;int* a=nullptr;int* b=nullptr;float* distance=nullptr;bool failed=false;};
extern "C" void edt_pair_destroy(void* pointer) noexcept {
    auto* p=static_cast<Pair*>(pointer);if(!p)return;
    if(p->mask)cudaFree(p->mask);if(p->a)cudaFree(p->a);if(p->b)cudaFree(p->b);if(p->distance)cudaFree(p->distance);delete p;
}
extern "C" void* edt_pair_create(int nx,int ny,int nz) noexcept {
    if(nx<1||ny<1||nz<1||nx>512||ny>512||nz>512||static_cast<int64_t>(nx)*ny*nz>1048576)return nullptr;
    Pair* p=nullptr;
    try {p=new Pair{nx,ny,nz,nx*ny*nz};
        if(cudaMalloc(&p->mask,p->size)!=cudaSuccess || cudaMalloc(&p->a,2*p->size*sizeof(int))!=cudaSuccess ||
           cudaMalloc(&p->b,2*p->size*sizeof(int))!=cudaSuccess || cudaMalloc(&p->distance,p->size*sizeof(float))!=cudaSuccess) {
            edt_pair_destroy(p);return nullptr;
        }
        return p;
    }catch(...){edt_pair_destroy(p);return nullptr;}
}
extern "C" int edt_pair_query(void* pointer,const uint8_t* mask,double pitch,float* output,int32_t* squared) noexcept {
    auto* p=static_cast<Pair*>(pointer);
    if(!p||!mask||!output||!std::isfinite(pitch)||pitch<1e-12||pitch>1e12)return 1;
    if(p->failed)return 2;
    int ones=0;for(int i=0;i<p->size;++i){if(mask[i]>1)return 1;ones+=mask[i];}if(ones==0||ones==p->size)return 1;
    #define CHECK(x) if((x)!=cudaSuccess){p->failed=true;return 2;}
    CHECK(cudaMemcpy(p->mask,mask,p->size,cudaMemcpyHostToDevice));
    int sentinel=(p->nx-1)*(p->nx-1)+(p->ny-1)*(p->ny-1)+(p->nz-1)*(p->nz-1)+1;
    initialise<<<(p->size+255)/256,256>>>(p->mask,p->a,p->size,sentinel);CHECK(cudaGetLastError());
    int lengths[]={p->nx,p->ny,p->nz},strides[]={p->ny*p->nz,p->nz,1};int* a=p->a;int* b=p->b;
    for(int axis=0;axis<3;++axis){
        axis_envelope<<<(2*p->size/lengths[axis]+127)/128,128>>>(a,b,p->size,strides[axis],lengths[axis]);
        CHECK(cudaGetLastError());std::swap(a,b);
    }
    signed_output<<<(p->size+255)/256,256>>>(a,p->distance,p->size,pitch);CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(output,p->distance,p->size*sizeof(float),cudaMemcpyDeviceToHost));
    if(squared){CHECK(cudaMemcpy(squared,a,2*p->size*sizeof(int),cudaMemcpyDeviceToHost));}
    return 0;
    #undef CHECK
}
