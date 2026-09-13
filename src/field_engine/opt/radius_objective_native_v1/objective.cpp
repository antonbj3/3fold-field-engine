#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace {
constexpr int max_controls = 64;
bool valid(int n, int corners, int obstacle, const double* lengths,
           const std::int32_t* indices, const double* sdf,
           const double* p, const double* r, double* out) {
    if (n < 2 || n > max_controls || corners < 0 || corners > max_controls ||
        (obstacle != 0 && obstacle != 1) || !lengths || !p || !r || !out ||
        (corners && !indices) || (obstacle && !sdf)) return false;
    for (int i = 0; i < n; ++i)
        if (!std::isfinite(r[i]) || r[i] <= 0 ||
            (obstacle && !std::isfinite(sdf[i]))) return false;
    for (int i = 0; i < n - 1; ++i)
        if (!std::isfinite(lengths[i]) || lengths[i] < 0) return false;
    for (int i = 0; i < corners; ++i)
        if (indices[i] < 0 || indices[i] >= n) return false;
    for (int i = 0; i < 10; ++i) if (!std::isfinite(p[i])) return false;
    return p[1] > 0 && p[2] > 0 && p[3] > 0 && p[7] > 0 && p[8] > 0;
}

// Keep each scalar operation and each accumulation in the Python reference order.
void evaluate(int n, int corners, int obstacle, const double* lengths,
              const std::int32_t* indices, const double* sdf,
              const double* p, const double* r, double* out) {
    const double kb=p[0], flow=p[1], rho=p[2], mu=p[3], ka=p[4], ki=p[5];
    const double area_min=p[6], area_den=p[7], intrusion_den=p[8], pi=p[9];
    double friction=0.0;
    for (int i=0; i<n-1; ++i) {
        const double avg=0.5*(r[i]+r[i+1]);
        const double area=pi*std::pow(avg,2.0), dh=2*avg;
        const double area_m2=area*1e-6, dh_m=dh*1e-3, ds_m=lengths[i]*1e-3;
        const double v=flow/std::max(area_m2,1e-9);
        const double re=rho*v*dh_m/mu;
        const double fd=re<2300 ? 64.0/re : 0.316*std::pow(re,-0.25);
        const double dp=fd*(ds_m/std::max(dh_m,1e-9))*0.5*rho*std::pow(v,2.0);
        friction+=dp;
    }
    double bend=0.0;
    for (int j=0; j<corners; ++j) {
        const double area=pi*std::pow(r[indices[j]],2.0);
        const double v=flow/std::max(area*1e-6,1e-9);
        bend+=kb*0.5*rho*std::pow(v,2.0);
    }
    double pen_area=0.0;
    for (int i=0; i<n; ++i) {
        const double a=pi*std::pow(r[i],2.0);
        const double viol=std::max(0.0,area_min-a);
        pen_area+=ka*std::pow(viol/area_den,2.0);
    }
    double pen_intrude=0.0, max_viol=0.0;
    if (obstacle) for (int i=0; i<n; ++i) {
        const double viol=std::max(0.0,r[i]-sdf[i]);
        max_viol=std::max(max_viol,viol);
        pen_intrude+=ki*std::pow(viol/intrusion_den,2.0);
    }
    out[0]=friction+bend+pen_area+pen_intrude;
    out[1]=friction;out[2]=bend;out[3]=pen_area;out[4]=pen_intrude;
    out[5]=friction+bend;out[6]=corners;out[7]=max_viol;
}
}

extern "C" int field_radius_evaluate(int n, int corners, int obstacle,
        const double* lengths, const std::int32_t* indices, const double* sdf,
        const double* params, const double* radii, double* output, std::int64_t capacity) {
    if (capacity<8 || !valid(n,corners,obstacle,lengths,indices,sdf,params,radii,output)) return 1;
    evaluate(n,corners,obstacle,lengths,indices,sdf,params,radii,output);
    return 0;
}

extern "C" int field_radius_central(int n, int corners, int obstacle,
        const double* lengths, const std::int32_t* indices, const double* sdf,
        const double* params, const double* radii, double* output, std::int64_t capacity,
        double h) {
    if (!std::isfinite(h) || h<=0 || n<2 || n>max_controls || capacity<16*n ||
        !valid(n,corners,obstacle,lengths,indices,sdf,params,radii,output)) return 1;
    for (int k=0; k<n; ++k) if (!std::isfinite(radii[k]+h)) return 1;
    double scratch[max_controls];
    std::memcpy(scratch,radii,n*sizeof(double));
    for (int k=0; k<n; ++k) {
        scratch[k]=radii[k]+h;
        evaluate(n,corners,obstacle,lengths,indices,sdf,params,scratch,output+16*k);
        scratch[k]=std::max(1.0,radii[k]-h);
        evaluate(n,corners,obstacle,lengths,indices,sdf,params,scratch,output+16*k+8);
        scratch[k]=radii[k];
    }
    return 0;
}
