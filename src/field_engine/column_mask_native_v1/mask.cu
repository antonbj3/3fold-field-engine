#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <algorithm>
__global__ void classify(const double* keys,const int* signs,const int* offsets,const double* fractions,int nz,int size,uint8_t* mask){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=size)return;
 int column=i/nz;double query=double(column)+fractions[i%nz];int sum=0;
 for(int j=offsets[column];j<offsets[column+1];++j)if(keys[j]>query)sum+=signs[j];
 mask[i]=sum!=0;
}
__global__ void boundary(const uint8_t* mask,uint8_t* surface,int nx,int ny,int nz){
 int i=blockIdx.x*blockDim.x+threadIdx.x,size=nx*ny*nz;if(i>=size)return;
 int z=i%nz,y=(i/nz)%ny,x=i/(ny*nz);
 surface[i]=mask[i] && (x==0 || x==nx-1 || y==0 || y==ny-1 || z==0 || z==nz-1 ||
 !mask[i-ny*nz] || !mask[i+ny*nz] || !mask[i-nz] || !mask[i+nz] || !mask[i-1] || !mask[i+1]);
}
extern "C" int column_mask_surface(const double* keys,const int32_t* signs,int hits,const int32_t* offsets,const double* fractions,int nx,int ny,int nz,uint8_t* mask,uint8_t* surface){
 if(!keys || !signs || !offsets || !fractions || !mask || !surface || hits<0 || hits>1000000 || nx<1 || ny<1 || nz<1 || nx>512 || ny>512 || nz>512 || nx*ny*nz>1048576)return 1;
 int columns=nx*ny,size=columns*nz;
 if(offsets[0]!=0 || offsets[columns]!=hits)return 1;
 for(int i=0;i<columns;++i)if(offsets[i]<0 || offsets[i]>offsets[i+1] || offsets[i+1]>hits)return 1;
 for(int i=0;i<hits;++i)if(!std::isfinite(keys[i]) || (signs[i]!=1 && signs[i]!=-1))return 1;
 for(int i=0;i<nz;++i)if(!std::isfinite(fractions[i]) || fractions[i]<0 || fractions[i]>.999)return 1;
 double *dk=nullptr,*df=nullptr;int *ds=nullptr,*di=nullptr;uint8_t *dm=nullptr,*db=nullptr;int status=2;
 #define CHECK(x) if((x)!=cudaSuccess)throw 2
 try{
  CHECK(cudaMalloc(&dk,std::max(hits,1)*sizeof(double)));CHECK(cudaMalloc(&ds,std::max(hits,1)*sizeof(int)));
  CHECK(cudaMalloc(&di,(columns+1)*sizeof(int)));CHECK(cudaMalloc(&df,nz*sizeof(double)));CHECK(cudaMalloc(&dm,size));CHECK(cudaMalloc(&db,size));
  if(hits){CHECK(cudaMemcpy(dk,keys,hits*sizeof(double),cudaMemcpyHostToDevice));CHECK(cudaMemcpy(ds,signs,hits*sizeof(int),cudaMemcpyHostToDevice));}
  CHECK(cudaMemcpy(di,offsets,(columns+1)*sizeof(int),cudaMemcpyHostToDevice));CHECK(cudaMemcpy(df,fractions,nz*sizeof(double),cudaMemcpyHostToDevice));
  classify<<<(size+255)/256,256>>>(dk,ds,di,df,nz,size,dm);CHECK(cudaGetLastError());
  boundary<<<(size+255)/256,256>>>(dm,db,nx,ny,nz);CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(mask,dm,size,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(surface,db,size,cudaMemcpyDeviceToHost));status=0;
 }catch(...){status=2;}
 for(void* p:{static_cast<void*>(dk),static_cast<void*>(ds),static_cast<void*>(di),static_cast<void*>(df),static_cast<void*>(dm),static_cast<void*>(db)})if(p && cudaFree(p)!=cudaSuccess)status=2;
 return status;
}
