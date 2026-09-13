#pragma once
#include <optix.h>
#include <cuda_runtime.h>
struct WindingParams { OptixTraversableHandle gas; const double3* corners; const float3* rays;
 const double2* queries; int* counts; double* depths; int* signs; unsigned capacity; double area_floor; const float4* interval_corners; const float4* interval_queries; };
