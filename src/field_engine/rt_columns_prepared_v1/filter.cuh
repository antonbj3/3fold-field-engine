#pragma once
#include <cuda_runtime.h>
// Bounds enclose the frozen double subtraction/multiply/subtract sequence.
// Compile with --fmad=false --ftz=false. Uncertain signs always fall through.
struct Interval {float lo,hi;};
__device__ Interval multiply(Interval a,Interval b){
    float lo=fminf(fminf(__fmul_rd(a.lo,b.lo),__fmul_rd(a.lo,b.hi)),fminf(__fmul_rd(a.hi,b.lo),__fmul_rd(a.hi,b.hi)));
    float hi=fmaxf(fmaxf(__fmul_ru(a.lo,b.lo),__fmul_ru(a.lo,b.hi)),fmaxf(__fmul_ru(a.hi,b.lo),__fmul_ru(a.hi,b.hi)));
    return {lo,hi};
}
__device__ Interval determinant(Interval ax,Interval ay,Interval bx,Interval by){
    Interval left=multiply(ax,by),right=multiply(ay,bx);
    return {__fsub_rd(left.lo,right.hi),__fsub_ru(left.hi,right.lo)};
}
__device__ void interval_numerators(float4 a,float4 b,float4 c,float4 q,Interval* result){
    Interval ax={__fsub_rd(a.x,q.y),__fsub_ru(a.y,q.x)},ay={__fsub_rd(a.z,q.w),__fsub_ru(a.w,q.z)};
    Interval bx={__fsub_rd(b.x,q.y),__fsub_ru(b.y,q.x)},by={__fsub_rd(b.z,q.w),__fsub_ru(b.w,q.z)};
    Interval cx={__fsub_rd(c.x,q.y),__fsub_ru(c.y,q.x)},cy={__fsub_rd(c.z,q.w),__fsub_ru(c.w,q.z)};
    result[0]=determinant(bx,by,cx,cy);result[1]=determinant(cx,cy,ax,ay);result[2]=determinant(ax,ay,bx,by);
}
__device__ bool interval_outside(float4 a,float4 b,float4 c,float4 q){
    Interval result[3];interval_numerators(a,b,c,q,result);
    return (result[0].lo>0||result[1].lo>0||result[2].lo>0)&&(result[0].hi<0||result[1].hi<0||result[2].hi<0);
}
