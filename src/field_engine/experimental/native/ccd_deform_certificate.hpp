#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
namespace deform_ccd {
using V=Eigen::Vector3d;
using X=Eigen::MatrixXd;
using F=Eigen::MatrixXi;
inline double down(double v){return std::nextafter(v,-INFINITY);}
inline double up(double v){return std::nextafter(v,INFINITY);}
struct I{double lo,hi; I(double a=0):lo(a),hi(a){} I(double a,double b):lo(a),hi(b){} };
inline I operator+(I a,I b){return {down(a.lo+b.lo),up(a.hi+b.hi)};}
inline I operator-(I a,I b){return {down(a.lo-b.hi),up(a.hi-b.lo)};}
inline I operator*(I a,I b){double v[]={a.lo*b.lo,a.lo*b.hi,a.hi*b.lo,a.hi*b.hi};return {down(*std::min_element(v,v+4)),up(*std::max_element(v,v+4))};}
inline I operator/(I a,I b){if(b.lo<=0&&b.hi>=0)return {-INFINITY,INFINITY};return a*I(down(1/b.hi),up(1/b.lo));}
inline I square(I a){if(a.lo<=0&&a.hi>=0)return {0,up(std::max(a.lo*a.lo,a.hi*a.hi))};return {down(std::min(a.lo*a.lo,a.hi*a.hi)),up(std::max(a.lo*a.lo,a.hi*a.hi))};}
inline I dot(const V&a,const V&b){I s;for(int k=0;k<3;k++)s=s+I(a[k])*I(b[k]);return s;}
inline double normup(const V&a){I s;for(int k=0;k<3;k++)s=s+square(I(a[k]));return up(std::sqrt(std::max(0.,s.hi)));}
inline V closest(const V&a,const V&b,const V&c){ // proposal only; soundness never depends on closeness
 V ab=b-a,ac=c-a,ap=-a;double d1=ab.dot(ap),d2=ac.dot(ap);if(d1<=0&&d2<=0)return a;
 V bp=-b;double d3=ab.dot(bp),d4=ac.dot(bp);if(d3>=0&&d4<=d3)return b;
 double vc=d1*d4-d3*d2;if(vc<=0&&d1>=0&&d3<=0)return a+(d1/(d1-d3))*ab;
 V cp=-c;double d5=ab.dot(cp),d6=ac.dot(cp);if(d6>=0&&d5<=d6)return c;
 double vb=d5*d2-d1*d6;if(vb<=0&&d2>=0&&d6<=0)return a+(d2/(d2-d6))*ac;
 double va=d3*d6-d5*d4;if(va<=0&&(d4-d3)>=0&&(d5-d6)>=0)return b+((d4-d3)/((d4-d3)+(d5-d6)))*(c-b);
 double den=va+vb+vc;if(!std::isfinite(den)||den==0)return (a+b+c)/3.;return a+(vb/den)*ab+(vc/den)*ac;
}
struct Family{X x0;std::vector<X> x1;double box=0;bool complete=false;};
struct Certificate{double prefix=0;bool licensed=false;size_t faces=0;const char*reason="declined";};
class Transport;
class Panel {
 friend class Transport;
 F f_;int nv_;double radius_;std::vector<std::array<I,4>> inv_;
public:
 Panel(const X&rest,const F&faces,const F&edges,double radius):f_(faces),nv_(rest.rows()),radius_(radius){
 if(rest.cols()!=3||rest.rows()<4||!rest.allFinite()||faces.cols()!=3||edges.cols()!=2||!std::isfinite(radius)||radius<=0)throw std::invalid_argument("shape/radius");
 std::set<double> xs,ys;std::map<std::pair<double,double>,int> lookup;
 for(int i=0;i<nv_;i++){xs.insert(rest(i,0));ys.insert(rest(i,1));if(!lookup.emplace(std::make_pair(rest(i,0),rest(i,1)),i).second)throw std::invalid_argument("reference duplicates");}
 if(xs.size()<2||ys.size()<2||xs.size()*ys.size()!=size_t(nv_))throw std::invalid_argument("reference grid");
 using T=std::array<int,3>;using E=std::array<int,2>;auto tri=[](int a,int b,int c){T t{a,b,c};std::sort(t.begin(),t.end());return t;};auto edge=[](int a,int b){return E{std::min(a,b),std::max(a,b)};};
 std::set<T> given,expected;std::set<E> induced,es;std::vector<double>x(xs.begin(),xs.end()),y(ys.begin(),ys.end());
 for(size_t i=0;i+1<x.size();i++)for(size_t j=0;j+1<y.size();j++){int a=lookup.at({x[i],y[j]}),b=lookup.at({x[i+1],y[j]}),c=lookup.at({x[i],y[j+1]}),d=lookup.at({x[i+1],y[j+1]});expected.insert(tri(a,b,c));expected.insert(tri(b,d,c));}
 for(int i=0;i<faces.rows();i++){int a=faces(i,0),b=faces(i,1),c=faces(i,2);if(a<0||b<0||c<0||a>=nv_||b>=nv_||c>=nv_||a==b||a==c||b==c||!given.insert(tri(a,b,c)).second)throw std::invalid_argument("faces");for(E e:{edge(a,b),edge(a,c),edge(b,c)})induced.insert(e);
 I ux=I(rest(b,0))-I(rest(a,0)),uy=I(rest(b,1))-I(rest(a,1)),vx=I(rest(c,0))-I(rest(a,0)),vy=I(rest(c,1))-I(rest(a,1)),det=ux*vy-uy*vx;if(det.lo<=0&&det.hi>=0)throw std::invalid_argument("reference determinant");inv_.push_back({vy/det,(I(0)-vx)/det,(I(0)-uy)/det,ux/det});}
 for(int i=0;i<edges.rows();i++){int a=edges(i,0),b=edges(i,1);if(a<0||b<0||a>=nv_||b>=nv_||a==b||!es.insert(edge(a,b)).second)throw std::invalid_argument("edges");}
 if(given!=expected||es!=induced)throw std::invalid_argument("complete topology");
 }
 Panel& operator=(const Panel&)=delete;
 Panel& operator=(Panel&&)=delete;
 const F& faces()const{return f_;}double radius()const{return radius_;}
 bool shape(const Family&q)const{if(!q.complete||q.x1.empty()||q.box<0||!std::isfinite(q.box)||q.x0.rows()!=nv_||q.x0.cols()!=3||!q.x0.allFinite()||q.x0.cwiseAbs().maxCoeff()>1e6)return false;for(auto&x:q.x1)if(x.rows()!=nv_||x.cols()!=3||!x.allFinite()||x.cwiseAbs().maxCoeff()>1e6)return false;return true;}
 bool injection(const Family&q)const{
 if(!shape(q))return false;
 for(size_t bi=0;bi<=q.x1.size();bi++){const X&x=bi?q.x1[bi-1]:q.x0;
 for(int i=0;i<f_.rows();i++){int a=f_(i,0),b=f_(i,1),c=f_(i,2);I g[4];for(int k=0;k<2;k++){auto coord=[&](int j){return I(down(x(j,k)-q.box),up(x(j,k)+q.box));};I u=coord(b)-coord(a),v=coord(c)-coord(a);g[2*k]=u*inv_[i][0]+v*inv_[i][2];g[2*k+1]=u*inv_[i][1]+v*inv_[i][3];}
 I off=(g[1]+g[2])*I(.5);I det=g[0]*g[3]-square(off);if(!(g[0].lo>0&&g[3].lo>0&&det.lo>0))return false;}}
 return true;
 }
private:
 Certificate support_impl(const Family&q,bool check_injection)const{
 Certificate out;if(!shape(q)){out.reason="incomplete_or_invalid";return out;}if(check_injection&&!injection(q)){out.reason="projection";return out;}out.licensed=true;out.prefix=1;out.reason="panel_sphere_support";
 for(int i=0;i<f_.rows();i++){V n=closest(q.x0.row(f_(i,0)),q.x0.row(f_(i,1)),q.x0.row(f_(i,2)));if(!n.allFinite()){out.prefix=0;out.reason="direction";return out;}
 double nr=normup(n);I support=I(radius_)*I(nr);I err;for(int k=0;k<3;k++)err=err+I(q.box)*I(std::abs(n[k]));support=support+err;
 double g0=INFINITY,g1=INFINITY;for(int j=0;j<3;j++){int v=f_(i,j);g0=std::min(g0,(dot(n,q.x0.row(v))-support).lo);for(auto&x:q.x1)g1=std::min(g1,(dot(n,x.row(v))-support).lo);}out.faces++;
 if(!std::isfinite(g0)||!std::isfinite(g1)||!std::isfinite(nr)){out.prefix=0;out.reason="arithmetic";return out;}
 if(!(g0>0)){out.prefix=0;out.reason="initial_support";return out;}if(g1<=0){I root=I(g0)/(I(g0)-I(g1));out.prefix=std::min(out.prefix,std::max(0.,down(root.lo-1e-12)));}}
 return out;
 }
public:
 Certificate support(const Family&q)const{return support_impl(q,true);}
 double gradient_displacement_constant()const{double bound=0;for(auto&r:inv_){double c0=up(std::max(std::abs(r[0].lo),std::abs(r[0].hi))+std::max(std::abs(r[2].lo),std::abs(r[2].hi))),c1=up(std::max(std::abs(r[1].lo),std::abs(r[1].hi))+std::max(std::abs(r[3].lo),std::abs(r[3].hi)));I z=(square(I(c0))+square(I(c1)))*I(8);bound=std::max(bound,up(std::sqrt(z.hi)));}return bound;}
 double injection_margin(const X&x)const{Family q{x,{x},0,true};if(!shape(q))return 0;double margin=INFINITY;for(int i=0;i<f_.rows();i++){int a=f_(i,0),b=f_(i,1),c=f_(i,2);I g[4];for(int k=0;k<2;k++){I u=I(x(b,k))-I(x(a,k)),v=I(x(c,k))-I(x(a,k));g[2*k]=u*inv_[i][0]+v*inv_[i][2];g[2*k+1]=u*inv_[i][1]+v*inv_[i][3];}I off=(g[1]+g[2])*I(.5),det=g[0]*g[3]-square(off),trace=g[0]+g[3];if(!(g[0].lo>0&&g[3].lo>0&&det.lo>0&&trace.lo>0))return 0;double z=(I(det.lo)/I(trace.hi)).lo;if(!std::isfinite(z))return 0;margin=std::min(margin,z);}return margin;}
 double clearance_margin(const X&x)const{double gap=INFINITY;for(int i=0;i<f_.rows();i++){V n=closest(x.row(f_(i,0)),x.row(f_(i,1)),x.row(f_(i,2)));double nr=normup(n);if(!(nr>0)||!std::isfinite(nr))return 0;for(int j=0;j<3;j++){double g=(dot(n,x.row(f_(i,j)))-I(radius_)*I(nr)).lo;g=(I(g)/I(nr)).lo;if(!std::isfinite(g))return 0;gap=std::min(gap,g);}}return std::max(0.,gap);}
 // Conventional distance/speed conservative advancement, same graph license.
 Certificate advancement(const Family&q)const{
 Certificate out;if(!shape(q)||!injection(q)){out.reason="projection";return out;}out.licensed=true;out.prefix=1;out.reason="distance_speed";
 for(int i=0;i<f_.rows();i++){V n=closest(q.x0.row(f_(i,0)),q.x0.row(f_(i,1)),q.x0.row(f_(i,2)));double nr=normup(n);if(!(nr>0)){out.prefix=0;return out;}
 I s=I(radius_)*I(nr);for(int k=0;k<3;k++)s=s+I(q.box)*I(std::abs(n[k]));double g=INFINITY,speed=0;for(int j=0;j<3;j++){int v=f_(i,j);g=std::min(g,(dot(n,q.x0.row(v))-s).lo);for(auto&x:q.x1){V d=x.row(v)-q.x0.row(v); // conservative componentwise subtraction envelope
 I ss;for(int k=0;k<3;k++){I dd=I(x(v,k))-I(q.x0(v,k));dd=dd+I(-2*q.box,2*q.box);ss=ss+square(dd);}speed=std::max(speed,up(std::sqrt(std::max(0.,ss.hi))));}}
 if(!(g>0)||!std::isfinite(g)||!std::isfinite(speed)||!std::isfinite(nr)){out.prefix=0;return out;}double gap=(I(g)/I(nr)).lo;if(speed>0&&gap<=speed)out.prefix=std::min(out.prefix,std::max(0.,down((I(gap)/I(speed)).lo-1e-12)));out.faces++;}
 return out;
 }
};
// Both transport arms use the conventional Hausdorff/Lipschitz license.
class Transport {
 const Panel&p_;X anchor_;double injection_=0,clearance_=0,K_;bool have_=false;
 std::pair<double,double> displacement(const Family&q)const{double planar=0,spatial=0;for(size_t bi=0;bi<=q.x1.size();bi++){const X&x=bi?q.x1[bi-1]:q.x0;for(int i=0;i<x.rows();i++){I ss;for(int k=0;k<3;k++){I d=I(x(i,k))-I(anchor_(i,k));double a=up(std::max(std::abs(d.lo),std::abs(d.hi))+q.box);if(k<2)planar=std::max(planar,a);ss=ss+square(I(a));}spatial=std::max(spatial,up(std::sqrt(ss.hi)));}}return {planar,spatial};}
 bool licensed(double planar)const{return (I(K_)*I(planar)).hi<injection_;}
public:
 // A full-step consumer must supply its sound original CCD reserve. Invalid
 // families cannot be repaired by checking only their supplied endpoints.
 template<class Fallback>
 Certificate run_or_fallback(const Family&q, Fallback&& reserve){
  if(!p_.shape(q))return {0,false,0,"invalid_family"};
  Certificate c=run(q);
  if(c.licensed&&c.prefix==1)return c;
  c=std::forward<Fallback>(reserve)(q);
  if(!c.licensed||!std::isfinite(c.prefix)||c.prefix<0||c.prefix>1)
   return {0,false,0,"invalid_reserve"};
  return c;
 }
 size_t hits=0,refreshes=0,prefix_queries=0;
 Transport(Panel&&)=delete;
 Transport(const Panel&&)=delete;
 explicit Transport(const Panel&p):p_(p),K_(p.gradient_displacement_constant()){}
 Certificate run(const Family&q){Certificate out;if(!p_.shape(q)){out.reason="invalid_family";return out;}
 if(have_){auto d=displacement(q);if(licensed(d.first)&&d.second<clearance_){hits++;return {1,true,0,"transported_clearance"};}}
 anchor_=q.x0;injection_=p_.injection_margin(anchor_);clearance_=p_.clearance_margin(anchor_);have_=injection_>0;refreshes++;auto d=displacement(q);bool self=have_&&licensed(d.first);
 if(self&&d.second<clearance_){hits++;return {1,true,0,"refreshed_clearance"};}
 if(!self){out.reason="projection_transport";return out;}
 prefix_queries++;return p_.support_impl(q,false);
 }
};
}
