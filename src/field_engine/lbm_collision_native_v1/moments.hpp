#pragma once
#include <cmath>
#include <cstdint>
#include <limits>
#include <cstring>
#include <utility>

namespace field_lbm_detail {
inline constexpr double stencil[19][3]={
    {0.0,0.0,0.0},
    {1.0,0.0,0.0},
    {-1.0,0.0,0.0},
    {0.0,1.0,0.0},
    {0.0,-1.0,0.0},
    {0.0,0.0,1.0},
    {0.0,0.0,-1.0},
    {1.0,1.0,0.0},
    {-1.0,-1.0,0.0},
    {1.0,-1.0,0.0},
    {-1.0,1.0,0.0},
    {1.0,0.0,1.0},
    {-1.0,0.0,-1.0},
    {1.0,0.0,-1.0},
    {-1.0,0.0,1.0},
    {0.0,1.0,1.0},
    {0.0,-1.0,-1.0},
    {0.0,1.0,-1.0},
    {0.0,-1.0,1.0}
};
template<std::size_t Q> inline void add_moment(const double* f, double* m) {
    m[0]+=f[Q]*stencil[Q][0];
    m[1]+=f[Q]*stencil[Q][1];
    m[2]+=f[Q]*stencil[Q][2];
}
template<std::size_t... Q> inline void static_moments(const double* f, double* m,
                                                    std::index_sequence<Q...>) {
    (add_moment<Q>(f,m),...);
}
template<std::size_t Q> inline void project(const double* u, double* cu) {
    double value=0.0;
    value+=u[0]*stencil[Q][0];
    value+=u[1]*stencil[Q][1];
    value+=u[2]*stencil[Q][2];
    cu[Q]=value;
}
template<std::size_t... Q> inline void static_projections(const double* u, double* cu,
                                                        std::index_sequence<Q...>) {
    (project<Q>(u,cu),...);
}
template<bool Canonical> inline int moments_loop(std::int64_t cells, const double* f,
        const std::uint8_t* solid, const double* force, const double* e,
        double* rho, double* u, double* cu, double* usq, double* ua) {
    for (std::int64_t i=0; i<cells; ++i) {
        const double* v=f+19*i;
        // NumPy's contiguous nineteen-value pairwise sum: eight accumulators,
        // the balanced reduction, then the three-element tail.
        double r0=v[0]+v[8], r1=v[1]+v[9], r2=v[2]+v[10], r3=v[3]+v[11];
        double r4=v[4]+v[12], r5=v[5]+v[13], r6=v[6]+v[14], r7=v[7]+v[15];
        double r=((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7));
        r+=v[16];r+=v[17];r+=v[18];
        r=r>1e-9 ? r : 1.0;rho[i]=r;
        double m[3]={0.0,0.0,0.0};
        if constexpr (Canonical) static_moments(v,m,std::make_index_sequence<19>{});
        else for (int q=0; q<19; ++q) for (int a=0; a<3; ++a) m[a]+=v[q]*e[3*q+a];
        for (int a=0; a<3; ++a) {
            const double velocity=(m[a]+(0.5*r)*force[3*i+a])/r;
            u[3*i+a]=solid[i] ? 0.0 : velocity;
        }
        if constexpr (Canonical) static_projections(u+3*i,cu+19*i,std::make_index_sequence<19>{});
        else for (int q=0; q<19; ++q) {
            double value=0.0;
            for (int a=0; a<3; ++a) value+=u[3*i+a]*e[3*q+a];
            cu[19*i+q]=value;
        }
        double squared=0.0, forced=0.0;
        for (int a=0; a<3; ++a) {
            squared+=u[3*i+a]*u[3*i+a];
            forced+=u[3*i+a]*force[3*i+a];
        }
        usq[i]=squared;ua[i]=forced;
    }
    return 0;
}
}

extern "C" int field_lbm_moments(std::int64_t cells, const double* f,
        const std::uint8_t* solid, const double* force, const double* e,
        double* rho, double* u, double* cu, double* usq, double* ua) {
    if (cells<0 || cells>std::numeric_limits<std::int64_t>::max()/19 ||
        (cells && (!f || !solid || !force || !e || !rho || !u || !cu || !usq || !ua))) return 1;
    if (cells==0) return 0;
    if (std::memcmp(e,field_lbm_detail::stencil,sizeof(field_lbm_detail::stencil))==0)
        return field_lbm_detail::moments_loop<true>(cells,f,solid,force,e,rho,u,cu,usq,ua);
    return field_lbm_detail::moments_loop<false>(cells,f,solid,force,e,rho,u,cu,usq,ua);
}
