#include "field_engine/experimental/native/ccd_scene_certificate.hpp"
#include <iostream>
#include <stdexcept>
using namespace scene_cert;
struct Fixture{Eigen::MatrixXd x,y;Eigen::MatrixXi e,f;};
Fixture make(int n=3,bool crossing=false,bool glancing=false){Fixture m;m.x.resize(2*n*n,3);m.y.resize(2*n*n,3);for(int layer=0;layer<2;layer++)for(int i=0;i<n;i++)for(int j=0;j<n;j++){int z=layer*n*n+i*n+j;double px=i*.5,py=j*.5,pz=layer*std::ldexp(1.,-10);if(glancing&&layer){px+=(n-1)*.5+std::ldexp(1.,-24)-1./16;pz=1./16;}m.x.row(z)<<px,py,pz;if(glancing){m.y.row(z)=m.x.row(z);if(layer){m.y(z,0)+=1./8;m.y(z,2)=-1./16;}}else{m.y.row(z)=m.x.row(z)+Eigen::RowVector3d(.5,0,.5);if(crossing){m.y.row(z)=m.x.row(z);if(layer)m.y(z,2)=-std::ldexp(1.,-10);}}}
std::vector<T>f;std::set<E>e;for(int l=0;l<2;l++)for(int i=0;i<n-1;i++)for(int j=0;j<n-1;j++){int a=l*n*n+i*n+j,b=a+n,c=a+1,d=b+1;f.push_back({a,b,c});f.push_back({b,d,c});}for(auto t:f)for(auto z:{edge(t[0],t[1]),edge(t[0],t[2]),edge(t[1],t[2])})e.insert(z);m.f.resize(f.size(),3);m.e.resize(e.size(),2);for(int i=0;i<(int)f.size();i++)for(int k=0;k<3;k++)m.f(i,k)=f[i][k];int i=0;for(auto z:e){m.e(i,0)=z[0];m.e(i++,1)=z[1];}return m;}
Result run(Fixture&m){return certify(m.x,m.y,m.e,m.f);}
int main(){int checks=0;auto check=[&](bool b,const char*n){if(!b)throw std::runtime_error(n);checks++;std::cout<<n<<" passed\n";};
{auto m=make();auto p=run(m);check(p.safe&&p.parts==2&&p.slabs==1,"parallel_full_scene");}
{auto m=make(3,false,true);auto p=run(m);check(p.safe&&p.slabs>1,"glancing_time_cover");}
{auto m=make(3,true);check(!run(m).safe,"actual_midstep_collision");}
{auto m=make();m.y(4,2)+=1./64;check(!run(m).safe,"deformed_vertex");}
{auto m=make();m.e.conservativeResize(m.e.rows()+1,2);m.e.row(m.e.rows()-1)<<0,8;check(!run(m).safe,"extra_nonmesh_edge");}
{auto m=make();m.f.row(0)=m.f.row(1);check(!run(m).safe,"duplicate_missing_face");}
{auto m=make();m.e.row(0)=m.e.row(1);check(!run(m).safe,"duplicate_missing_edge");}
{auto m=make();m.x(0,0)=1e-15;check(!run(m).safe,"nonlattice");}
{auto m=make();m.x(0,0)=std::numeric_limits<double>::infinity();check(!run(m).safe,"nonfinite");}
{auto m=make();m.x(0,0)=2048;check(!run(m).safe,"integer_bound");}
{auto m=make();m.f(0,0)=-1;check(!run(m).safe,"invalid_indices");}
{auto m=make();m.f(0,0)=m.f(0,1);check(!run(m).safe,"degenerate_face");}
{auto m=make();m.x.row(0)=m.x.row(1);m.y.row(0)=m.y.row(1);check(!run(m).safe,"duplicate_vertex_position");}
{auto m=make();for(int i=0;i<m.x.rows();i++){m.x(i,0)-=1024;m.y(i,0)-=1024;}check(run(m).safe,"negative_coordinate_boundary");}
{auto m=make();std::swap(m.f(0,0),m.f(0,2));check(run(m).safe,"triangle_orientation_invariance");}
{auto m=make();m.y.resize(m.y.rows()-1,3);check(!run(m).safe,"mismatched_input_shape");}
{auto m=make();for(int i=9;i<18;i++)m.x(i,2)=0;auto p=run(m);check(!p.safe&&std::string(p.reason)=="time_cover","initial_contact_refused");}
{auto m=make(3,false,true);for(int i=9;i<18;i++){m.x(i,0)-=std::ldexp(1.,-24);m.x(i,0)+=std::ldexp(1.,-40);m.y(i,0)-=std::ldexp(1.,-24);m.y(i,0)+=std::ldexp(1.,-40);}auto p=run(m);check(!p.safe&&std::string(p.reason)=="time_cover","depth_budget_is_decline");}
{auto m=make();std::vector<int>p(m.x.rows());std::iota(p.begin(),p.end(),0);std::reverse(p.begin(),p.end());auto z=m;for(int i=0;i<m.x.rows();i++){z.x.row(p[i])=m.x.row(i);z.y.row(p[i])=m.y.row(i);}for(int i=0;i<m.f.rows();i++)for(int k=0;k<3;k++)z.f(i,k)=p[m.f(i,k)];for(int i=0;i<m.e.rows();i++)for(int k=0;k<2;k++)z.e(i,k)=p[m.e(i,k)];check(run(z).safe,"vertex_index_permutation");}
std::cout<<"checks "<<checks<<"\n";}
