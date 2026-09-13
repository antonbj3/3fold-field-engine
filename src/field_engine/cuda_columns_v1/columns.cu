// Exact projected-column scan without OptiX initialization. Synchronous C ABI.
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

static void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Triangle { double3 a, b, c; double xmin, xmax, ymin, ymax; };
struct State {
    Triangle* triangles{}; double2* queries{}; int* counts{}; double* depths{}; int* signs{};
    uint32_t ntri{}, capacity{}, hits{}; double area_floor{};
    std::thread::id owner=std::this_thread::get_id();
    int release() noexcept {
        int status=0;
        for(void* p:{static_cast<void*>(triangles),static_cast<void*>(queries),static_cast<void*>(counts),
                     static_cast<void*>(depths),static_cast<void*>(signs)})
            if(p && cudaFree(p)!=cudaSuccess)status=2;
        return status;
    }
};

__global__ void scan_columns(const Triangle* triangles,uint32_t ntri,const double2* queries,
                             uint32_t count,uint32_t capacity,double area_floor,
                             int* counts,double* depths,int* signs) {
    uint32_t column=blockIdx.x*blockDim.x+threadIdx.x;
    if(column>=count)return;
    double2 q=queries[column];uint32_t found=0;
    for(uint32_t face=0;face<ntri;++face){
        const Triangle& triangle=triangles[face];
        if(q.x<triangle.xmin || q.x>triangle.xmax || q.y<triangle.ymin || q.y>triangle.ymax)continue;
        double3 a=triangle.a,b=triangle.b,c=triangle.c;
        double d=(b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
        if(!(fabs(d)>area_floor))continue;
        double l0=(b.x-q.x)*(c.y-q.y)-(b.y-q.y)*(c.x-q.x);
        double l1=(c.x-q.x)*(a.y-q.y)-(c.y-q.y)*(a.x-q.x);
        double l2=(a.x-q.x)*(b.y-q.y)-(a.y-q.y)*(b.x-q.x);
        int sign=(d>0)?1:-1;if(l0*sign<0 || l1*sign<0 || l2*sign<0)continue;
        double z=(l0*a.z+l1*b.z+l2*c.z)/d;
        if(found<capacity){size_t k=size_t(column)*capacity+found;depths[k]=z;signs[k]=sign;}
        ++found;
    }
    counts[column]=int(found);
}

extern "C" int columns_create(const char* backend,const double* input,uint32_t triangles,uint32_t capacity,
                               uint32_t hits,const double* anchor,void** handle) {
    if(!handle || *handle || !backend || std::strcmp(backend,"cuda-scan-v1") || !input || !anchor ||
       !triangles || triangles>1000000 || !capacity || capacity>262144 || !hits || hits>256)return 1;
    for(size_t i=0;i<size_t(triangles)*9;++i)if(!std::isfinite(input[i]) || std::abs(input[i])>1e12)return 1;
    for(int i=0;i<3;++i)if(!std::isfinite(anchor[i]) || std::abs(anchor[i])>1e12)return 1;
    State* state=nullptr;
    try{
        std::vector<Triangle> host(triangles);double scale=1.;
        for(size_t i=0;i<triangles;++i){
            const double* p=input+9*i;
            host[i]={make_double3(p[0],p[1],p[2]),make_double3(p[3],p[4],p[5]),make_double3(p[6],p[7],p[8]),
                     std::min({p[0],p[3],p[6]}),std::max({p[0],p[3],p[6]}),
                     std::min({p[1],p[4],p[7]}),std::max({p[1],p[4],p[7]})};
            for(int j=0;j<3;++j)scale=std::max({scale,std::abs(p[3*j]),std::abs(p[3*j+1])});
        }
        state=new State;state->ntri=triangles;state->capacity=capacity;state->hits=hits;
        state->area_floor=1e-12*scale*scale;
        check(cudaMalloc(&state->triangles,host.size()*sizeof(Triangle)));
        check(cudaMemcpy(state->triangles,host.data(),host.size()*sizeof(Triangle),cudaMemcpyHostToDevice));
        check(cudaMalloc(&state->queries,size_t(capacity)*sizeof(double2)));
        check(cudaMalloc(&state->counts,size_t(capacity)*sizeof(int)));
        check(cudaMalloc(&state->depths,size_t(capacity)*hits*sizeof(double)));
        check(cudaMalloc(&state->signs,size_t(capacity)*hits*sizeof(int)));
        *handle=state;return 0;
    }catch(...){if(state){state->release();delete state;}return 2;}
}

static int query(State& s,const double* xy,uint32_t count,int32_t* counts,double* depths,int32_t* signs,
                 uint32_t* width,bool compact) {
    if(s.owner!=std::this_thread::get_id())return 3;
    if(!xy || !counts || !depths || !signs || !count || count>s.capacity)return 1;
    for(size_t i=0;i<size_t(count)*2;++i)if(!std::isfinite(xy[i]) || std::abs(xy[i])>1e12)return 1;
    try{
        check(cudaMemcpy(s.queries,xy,size_t(count)*sizeof(double2),cudaMemcpyHostToDevice));
        scan_columns<<<(count+127)/128,128>>>(s.triangles,s.ntri,s.queries,count,s.hits,s.area_floor,s.counts,s.depths,s.signs);
        check(cudaGetLastError());check(cudaDeviceSynchronize());
        check(cudaMemcpy(counts,s.counts,size_t(count)*sizeof(int),cudaMemcpyDeviceToHost));
        uint32_t maximum=0;
        for(size_t i=0;i<count;++i){if(counts[i]>int(s.hits))return 4;maximum=std::max(maximum,uint32_t(counts[i]));}
        uint32_t columns=compact?maximum:s.hits;
        if(columns){
            check(cudaMemcpy2D(depths,columns*sizeof(double),s.depths,s.hits*sizeof(double),columns*sizeof(double),count,cudaMemcpyDeviceToHost));
            check(cudaMemcpy2D(signs,columns*sizeof(int),s.signs,s.hits*sizeof(int),columns*sizeof(int),count,cudaMemcpyDeviceToHost));
        }
        if(width)*width=maximum;
        return 0;
    }catch(...){return 2;}
}

extern "C" int columns_query(void* handle,const double* xy,uint32_t count,int32_t* counts,double* depths,int32_t* signs){
    if(!handle)return 1;return query(*static_cast<State*>(handle),xy,count,counts,depths,signs,nullptr,false);
}
extern "C" int columns_query_compact(void* handle,const double* xy,uint32_t count,int32_t* counts,double* depths,int32_t* signs,uint32_t* width){
    if(!handle || !width)return 1;return query(*static_cast<State*>(handle),xy,count,counts,depths,signs,width,true);
}
extern "C" int columns_destroy(void** handle){
    if(!handle)return 1;if(!*handle)return 0;State* s=static_cast<State*>(*handle);
    if(s->owner!=std::this_thread::get_id())return 3;
    int status=s->release();delete s;*handle=nullptr;return status;
}
