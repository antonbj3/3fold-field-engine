#include "../rt_columns_v1/params.h"
#include "filter.cuh"
#include <optix_device.h>
extern "C" { __constant__ WindingParams params; }
extern "C" __global__ void __raygen__winding(){
 unsigned i=optixGetLaunchIndex().x,count=0;
 optixTrace(params.gas,params.rays[i],make_float3(0,0,1),0.f,1.e16f,0.f,255,
 OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,0,1,0,count);params.counts[i]=int(count);
}
extern "C" __global__ void __miss__winding(){}
extern "C" __global__ void __intersection__columns(){
 unsigned face=optixGetPrimitiveIndex(),column=optixGetLaunchIndex().x;
 double3 a=params.corners[3*face],b=params.corners[3*face+1],c=params.corners[3*face+2];
 double2 q=params.queries[column];
 if(interval_outside(a,b,c,q))return;
 double d=(b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
 if(!(fabs(d)>params.area_floor))return;
 double l0=(b.x-q.x)*(c.y-q.y)-(b.y-q.y)*(c.x-q.x);
 double l1=(c.x-q.x)*(a.y-q.y)-(c.y-q.y)*(a.x-q.x);
 double l2=(a.x-q.x)*(b.y-q.y)-(a.y-q.y)*(b.x-q.x);
 int sign=(d>0)?1:-1;if(l0*sign<0 || l1*sign<0 || l2*sign<0)return;
 double z=(l0*a.z+l1*b.z+l2*c.z)/d;
 optixReportIntersection(1.f,0,__double2loint(z),__double2hiint(z),unsigned(sign));
}
extern "C" __global__ void __anyhit__winding(){
 unsigned n=optixGetPayload_0(),i=optixGetLaunchIndex().x;
 if(n<params.capacity){size_t k=size_t(i)*params.capacity+n;
 params.depths[k]=__hiloint2double(optixGetAttribute_1(),optixGetAttribute_0());
 params.signs[k]=int(optixGetAttribute_2());}
 optixSetPayload_0(n+1);optixIgnoreIntersection();
}
