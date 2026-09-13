#include <array>
#include <cassert>
#include <limits>
#include "../src/field_engine/lbm_collision_native_v1/moments.hpp"
int main() {
    std::array<double,38> f;f.fill(1.0);
    std::uint8_t solid[2]={0,1};double force[6]={};double e[57]={};
    struct Guard {double before;std::array<double,38> values;double after;};
    std::array<Guard,5> output;
    for (auto& g:output) {g.before=719;g.after=971;g.values.fill(-23);}
    auto r=output[0].values.data(), u=output[1].values.data(), cu=output[2].values.data();
    auto usq=output[3].values.data(), ua=output[4].values.data();
    assert(field_lbm_moments(2,f.data(),solid,force,e,r,u,cu,usq,ua)==0);
    assert(r[0]==19 && r[1]==19 && r[2]==-23);
    for (int i=0;i<6;++i) assert(u[i]==0);
    for (int i=0;i<38;++i) assert(cu[i]==0);
    assert(usq[0]==0 && usq[1]==0 && ua[0]==0 && ua[1]==0);
    assert(field_lbm_moments(2,f.data(),solid,force,&field_lbm_detail::stencil[0][0],r,u,cu,usq,ua)==0);
    assert(r[0]==19 && r[1]==19);
    for (int i=0;i<38;++i) assert(cu[i]==0);
    assert(field_lbm_moments(0,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr)==0);
    assert(field_lbm_moments(-1,f.data(),solid,force,e,r,u,cu,usq,ua)!=0);
    assert(field_lbm_moments(std::numeric_limits<std::int64_t>::max(),f.data(),solid,force,e,r,u,cu,usq,ua)!=0);
    assert(field_lbm_moments(2,nullptr,solid,force,e,r,u,cu,usq,ua)!=0);
    assert(field_lbm_moments(2,f.data(),nullptr,force,e,r,u,cu,usq,ua)!=0);
    assert(field_lbm_moments(2,f.data(),solid,nullptr,e,r,u,cu,usq,ua)!=0);
    assert(field_lbm_moments(2,f.data(),solid,force,nullptr,r,u,cu,usq,ua)!=0);
    assert(field_lbm_moments(2,f.data(),solid,force,e,nullptr,u,cu,usq,ua)!=0);
    assert(field_lbm_moments(2,f.data(),solid,force,e,r,nullptr,cu,usq,ua)!=0);
    assert(field_lbm_moments(2,f.data(),solid,force,e,r,u,nullptr,usq,ua)!=0);
    assert(field_lbm_moments(2,f.data(),solid,force,e,r,u,cu,nullptr,ua)!=0);
    assert(field_lbm_moments(2,f.data(),solid,force,e,r,u,cu,usq,nullptr)!=0);
    for (const auto& g:output) assert(g.before==719 && g.after==971);
    return 0;
}
