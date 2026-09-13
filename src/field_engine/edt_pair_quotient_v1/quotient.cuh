#pragma once
#include <cuda_runtime.h>
// EDT domain: |numerator| < 2^20, denominator in [2,1022]. Float division
// supplies only an estimate; integer inequalities determine the exact floor.
__device__ int corrected_floor(int numerator,int denominator){
    if(denominator==2)return numerator>=0?numerator/2:-((-numerator+1)/2);
    int result=__float2int_rd(__fdividef(static_cast<float>(numerator),static_cast<float>(denominator)));
    while(static_cast<long long>(result)*denominator>numerator)--result;
    while(static_cast<long long>(result+1)*denominator<=numerator)++result;
    return result;
}
