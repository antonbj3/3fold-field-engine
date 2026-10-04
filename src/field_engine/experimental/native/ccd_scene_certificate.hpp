#pragma once
// Experimental exact lattice/grid scene certificate; no external solver inside.
#include <Eigen/Core>
#include <array>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
namespace scene_cert {
using I=std::int64_t;using P=std::array<I,3>;using T=std::array<int,3>;using E=std::array<int,2>;
struct Part{std::vector<int> ids;std::vector<T> faces;};
struct Result{bool safe=false;int parts=0;int slabs=0;const char* reason="declined";};
inline bool lattice(const Eigen::MatrixXd&x,std::vector<P>&out){if(x.cols()!=3)return false;out.resize(x.rows());for(int i=0;i<x.rows();i++)for(int j=0;j<3;j++){double a=std::ldexp(x(i,j),40);if(!std::isfinite(a)||std::abs(a)>std::ldexp(1.,50)||std::floor(a)!=a)return false;out[i][j]=I(a);if(std::ldexp(double(out[i][j]),-40)!=x(i,j))return false;}return true;}
inline E edge(int a,int b){if(b<a)std::swap(a,b);return {a,b};}
inline T tri(int a,int b,int c){T z{a,b,c};std::sort(z.begin(),z.end());return z;}
inline bool grid(const Part&p,const std::vector<P>&x,const std::vector<P>&y){
 if(p.ids.size()<4||p.faces.empty())return false;const auto first=p.ids[0];P delta;for(int k=0;k<3;k++)delta[k]=y[first][k]-x[first][k];
 std::set<I> ax,ay;std::map<std::pair<I,I>,int> lookup;
 for(int i:p.ids){for(int k=0;k<3;k++)if(y[i][k]-x[i][k]!=delta[k])return false;if(x[i][2]!=x[first][2])return false;ax.insert(x[i][0]);ay.insert(x[i][1]);if(!lookup.emplace(std::make_pair(x[i][0],x[i][1]),i).second)return false;}
 if(ax.size()<2||ay.size()<2||ax.size()*ay.size()!=p.ids.size()||2*(ax.size()-1)*(ay.size()-1)!=p.faces.size())return false;
 std::set<T> faces;for(const auto&t:p.faces)if(!faces.insert(tri(t[0],t[1],t[2])).second)return false;
 std::vector<I> a(ax.begin(),ax.end()),b(ay.begin(),ay.end());
 for(size_t i=0;i+1<a.size();i++)for(size_t j=0;j+1<b.size();j++){
  int q=lookup.at({a[i],b[j]}),r=lookup.at({a[i+1],b[j]}),s=lookup.at({a[i],b[j+1]}),t=lookup.at({a[i+1],b[j+1]});
  if(!faces.count(tri(q,r,s))||!faces.count(tri(r,t,s)))return false;
 }
 return true;
}
// Exactly evaluate all vertex coordinates at dyadic time numerator/denominator.
inline bool slab(const Part&a,const Part&b,const std::vector<P>&x,const std::vector<P>&y,std::uint64_t lo,std::uint64_t hi,std::uint64_t den){
 for(int k=0;k<3;k++)for(int order=0;order<2;order++){
  bool sep=true;for(auto num:{lo,hi}){
   auto eval=[&](int i){return (__int128)(den-num)*x[i][k]+(__int128)num*y[i][k];};
   const auto&A=order?b:a;const auto&B=order?a:b;__int128 mx=eval(A.ids[0]),mn=eval(B.ids[0]);
   for(int i:A.ids)mx=std::max(mx,eval(i));for(int i:B.ids)mn=std::min(mn,eval(i));if(mx>=mn){sep=false;break;}
  }
  if(sep)return true;
 }
 return false;
}
inline bool cover(const Part&a,const Part&b,const std::vector<P>&x,const std::vector<P>&y,std::uint64_t lo,std::uint64_t hi,std::uint64_t den,int depth,int&visits,int&slabs){
 if(++visits>256)return false;if(slab(a,b,x,y,lo,hi,den)){++slabs;return true;}if(depth>=32)return false;
 // At a contact-free glancing pass only slabs near the axis switch refine.
 return cover(a,b,x,y,lo*2,lo+hi,den*2,depth+1,visits,slabs)&&cover(a,b,x,y,lo+hi,hi*2,den*2,depth+1,visits,slabs);
}
inline Result certify(const Eigen::MatrixXd&x0,const Eigen::MatrixXd&x1,const Eigen::MatrixXi&edges,const Eigen::MatrixXi&faces){
 Result out;if(x0.rows()!=x1.rows()||x0.rows()<4||edges.cols()!=2||faces.cols()!=3){out.reason="shape";return out;}
 std::vector<P>x,y;if(!lattice(x0,x)||!lattice(x1,y)){out.reason="lattice";return out;}
 const int n=x0.rows();std::set<E> induced,given;std::vector<int> parents(n);std::iota(parents.begin(),parents.end(),0);
 auto root=[&](int i){while(parents[i]!=i){parents[i]=parents[parents[i]];i=parents[i];}return i;};
 for(int i=0;i<faces.rows();i++){int a=faces(i,0),b=faces(i,1),c=faces(i,2);if(a<0||b<0||c<0||a>=n||b>=n||c>=n||a==b||b==c||a==c){out.reason="face_indices";return out;}for(auto z:{edge(a,b),edge(a,c),edge(b,c)}){induced.insert(z);int r=root(z[0]),s=root(z[1]);parents[s]=r;}}
 for(int i=0;i<edges.rows();i++){int a=edges(i,0),b=edges(i,1);if(a<0||b<0||a>=n||b>=n||a==b||!given.insert(edge(a,b)).second){out.reason="edge_indices";return out;}}
 if(given!=induced){out.reason="edge_set";return out;}
 std::map<int,int> labels;std::vector<Part> parts;for(int i=0;i<n;i++){int r=root(i);auto it=labels.emplace(r,labels.size()).first;if(it->second>=(int)parts.size())parts.emplace_back();parts[it->second].ids.push_back(i);}
 for(int i=0;i<faces.rows();i++)parts[labels.at(root(faces(i,0)))].faces.push_back({faces(i,0),faces(i,1),faces(i,2)});
 out.parts=parts.size();if(parts.size()>64){out.reason="component_budget";return out;}
 for(auto&p:parts)if(!grid(p,x,y)){out.reason="embedding_or_motion";return out;}
 for(size_t i=0;i<parts.size();i++)for(size_t j=0;j<i;j++){int visits=0;if(!cover(parts[i],parts[j],x,y,0,1,1,0,visits,out.slabs)){out.reason="time_cover";return out;}}
 out.safe=true;out.reason="grid_translation_time_cover";return out;
}
}
