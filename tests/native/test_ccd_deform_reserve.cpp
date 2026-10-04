#include "ccd_deform_certificate.hpp"
#include <cassert>
#include <iostream>
using namespace deform_ccd;
int main(){
 X x(4,3);x<<-1,-1,2,-1,1,2,1,-1,2,1,1,2;
 F f(2,3);f<<0,2,1,2,3,1;F e(5,2);e<<0,1,0,2,1,2,1,3,2,3;
 Panel panel(x,f,e,.25);Transport cache(panel);Family q{x,{x},0,true};
 int calls=0;
 auto stop=[&](const Family&family){calls++;assert(family.complete);return Certificate{0,true,0,"reserve_stopped"};};
 auto full=cache.run_or_fallback(q,stop);assert(full.prefix==1);assert(calls==0);
 // Loss of projection injectivity must invoke reserve and never mean full safe.
 q.x1[0].col(0)*=-1;auto folded=cache.run_or_fallback(q,stop);
 assert(calls==1);assert(folded.prefix==0);
 // A partial safe prefix also routes for this full-step consumer.
 q.x0.col(2).setConstant(.5);q.x1={q.x0};q.x1[0].col(2).setConstant(-.5);
 auto crossing=cache.run_or_fallback(q,stop);assert(calls==2);assert(crossing.prefix==0);
 // No omitted trajectory may be recovered by checking only supplied branches.
 q.complete=false;auto incomplete=cache.run_or_fallback(q,stop);
 assert(calls==2);assert(!incomplete.licensed);assert(incomplete.prefix==0);
 q.complete=true;
 for(double value:std::array<double,4>{NAN,INFINITY,-1.,2.}){
  auto bad=cache.run_or_fallback(q,[&](const Family&){return Certificate{value,true,0,"bad"};});
  assert(!bad.licensed);assert(bad.prefix==0);
 }
 auto unlicensed=cache.run_or_fallback(q,[&](const Family&){return Certificate{1,false,0,"bad"};});
 assert(!unlicensed.licensed);assert(unlicensed.prefix==0);
 bool threw=false;
 try{cache.run_or_fallback(q,[&](const Family&)->Certificate{throw std::runtime_error("reserve_failed");});}
 catch(const std::runtime_error&){threw=true;}
 assert(threw);
 // This 90 degree rotation violates the SPD sufficient test but has no contact:
 // static z=2 plane, proper rigid rotation, complete rectangular topology.
 q.x0=x;q.x0.col(0)=-x.col(1);q.x0.col(1)=x.col(0);q.x1={q.x0};
 assert(!panel.injection(q));
 auto analytic=[&](const Family&family){
  calls++;
  // Exact constant trajectory with unchanged proper topology and z=2 > radius.
  assert((family.x0-family.x1[0]).norm()==0);
  assert(family.x0.col(2).minCoeff()==2);
  return Certificate{1,true,0,"analytic_static_rigid_reserve"};
 };
 auto rotation=cache.run_or_fallback(q,analytic);assert(rotation.prefix==1);assert(rotation.licensed);assert(calls==3);
 std::cout<<"reserve regression groups passed\n";
}
