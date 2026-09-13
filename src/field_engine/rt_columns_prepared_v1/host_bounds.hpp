#pragma once
#include <cuda_runtime.h>
#include <cmath>
#include <limits>
inline float4 host_interval_point(double x,double y,const double3& anchor){
    float fx=static_cast<float>(x-anchor.x),fy=static_cast<float>(y-anchor.y);
    float infinity=std::numeric_limits<float>::infinity();
    return make_float4(std::nextafter(fx,-infinity),std::nextafter(fx,infinity),
                       std::nextafter(fy,-infinity),std::nextafter(fy,infinity));
}
