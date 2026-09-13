#pragma once
#include <cuda_runtime.h>
// Bounds enclose the frozen double subtraction/multiply/subtract sequence.
// Compile with --fmad=false --ftz=false. Uncertain signs always fall through.
struct Interval {float lo,hi;};
__device__ Interval enclose(double x){return {__double2float_rd(x),__double2float_ru(x)};}
__device__ Interval multiply(Interval a,Interval b){
    // Strict signs select the same extrema; zero-crossing intervals retain
    // the general four-product path, including its signed-zero behavior.
    if(a.lo>0.f){
        if(b.lo>0.f)return {__fmul_rd(a.lo,b.lo),__fmul_ru(a.hi,b.hi)};
        if(b.hi<0.f)return {__fmul_rd(a.hi,b.lo),__fmul_ru(a.lo,b.hi)};
    }else if(a.hi<0.f){
        if(b.lo>0.f)return {__fmul_rd(a.lo,b.hi),__fmul_ru(a.hi,b.lo)};
        if(b.hi<0.f)return {__fmul_rd(a.hi,b.hi),__fmul_ru(a.lo,b.lo)};
    }
    float lo=fminf(fminf(__fmul_rd(a.lo,b.lo),__fmul_rd(a.lo,b.hi)),fminf(__fmul_rd(a.hi,b.lo),__fmul_rd(a.hi,b.hi)));
    float hi=fmaxf(fmaxf(__fmul_ru(a.lo,b.lo),__fmul_ru(a.lo,b.hi)),fmaxf(__fmul_ru(a.hi,b.lo),__fmul_ru(a.hi,b.hi)));
    return {lo,hi};
}
__device__ Interval determinant(Interval ax,Interval ay,Interval bx,Interval by){
    Interval left=multiply(ax,by),right=multiply(ay,bx);
    return {__fsub_rd(left.lo,right.hi),__fsub_ru(left.hi,right.lo)};
}
__device__ void interval_numerators(double3 a,double3 b,double3 c,double2 q,Interval* result){
    Interval ax=enclose(a.x-q.x),ay=enclose(a.y-q.y);
    Interval bx=enclose(b.x-q.x),by=enclose(b.y-q.y);
    Interval cx=enclose(c.x-q.x),cy=enclose(c.y-q.y);
    result[0]=determinant(bx,by,cx,cy);result[1]=determinant(cx,cy,ax,ay);result[2]=determinant(ax,ay,bx,by);
}
__device__ bool interval_outside(double3 a,double3 b,double3 c,double2 q){
    Interval result[3];interval_numerators(a,b,c,q,result);
    return (result[0].lo>0||result[1].lo>0||result[2].lo>0)&&(result[0].hi<0||result[1].hi<0||result[2].hi<0);
}
