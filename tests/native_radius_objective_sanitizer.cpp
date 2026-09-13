#include <array>
#include <cassert>
#include <limits>
#include "../src/field_engine/opt/radius_objective_native_v1/objective.cpp"
int main() {
    std::array<double,63> lengths;lengths.fill(10);
    std::array<double,64> radii;radii.fill(20);
    std::array<double,64> sdf;sdf.fill(25);
    std::int32_t corner=0;
    double p[10]={.3,.01,1.2,1.8e-5,500,500,78.5,78.5,5,3.141592653589793};
    std::array<double,1026> buffer;buffer.fill(971);
    auto eval=[&](int n,int obstacle,const double* distance,const double* params,
                  std::int64_t capacity) {
        return field_radius_evaluate(n,1,obstacle,lengths.data(),&corner,distance,
                                     params,radii.data(),buffer.data()+1,capacity);
    };
    auto central=[&](int n,std::int64_t capacity,double h) {
        return field_radius_central(n,1,1,lengths.data(),&corner,sdf.data(),p,
                                    radii.data(),buffer.data()+1,capacity,h);
    };
    assert(eval(2,1,sdf.data(),p,8)==0);
    assert(buffer[0]==971 && buffer[9]==971 && buffer.back()==971);
    assert(central(64,1024,.5)==0);
    assert(buffer.front()==971 && buffer.back()==971);
    auto before=buffer;
    assert(eval(-1,1,sdf.data(),p,8)!=0);
    assert(eval(65,1,sdf.data(),p,8)!=0);
    assert(eval(2,1,sdf.data(),p,7)!=0);
    assert(eval(2,1,nullptr,p,8)!=0);
    assert(eval(2,1,sdf.data(),nullptr,8)!=0);
    corner=2;assert(eval(2,1,sdf.data(),p,8)!=0);corner=0;
    assert(central(64,1023,.5)!=0);
    assert(central(std::numeric_limits<int>::max(),1024,.5)!=0);
    assert(central(2,32,0)!=0);
    assert(central(2,32,-.5)!=0);
    assert(central(2,32,std::numeric_limits<double>::quiet_NaN())!=0);
    radii[0]=std::numeric_limits<double>::max();
    assert(central(2,32,std::numeric_limits<double>::max())!=0);
    radii[0]=20;
    assert(buffer==before);
    assert(eval(2,0,nullptr,p,8)==0);
    return 0;
}
