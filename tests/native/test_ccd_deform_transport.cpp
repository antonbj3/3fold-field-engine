#include "ccd_deform_certificate.hpp"
#include <cassert>
#include <iostream>
#include <type_traits>
using namespace deform_ccd;
static_assert(!std::is_copy_assignable_v<Panel>);
static_assert(!std::is_constructible_v<Transport,Panel&&>);
static_assert(std::is_same_v<decltype(&Panel::support),Certificate(Panel::*)(const Family&)const>);
int main(){X x(4,3);x<<-1,-1,2,-1,1,2,1,-1,2,1,1,2;F f(2,3);f<<0,2,1,2,3,1;F e(5,2);e<<0,1,0,2,1,2,1,3,2,3;Panel p(x,f,e,.25);Transport t(p),control(p);Family q{x,{x},0,true};assert(t.run(q).prefix==1);q.x0.col(2).array()-=.001;q.x1={q.x0};assert(t.run(q).prefix==1);assert(t.hits==2);assert(t.refreshes==1);
 q.x1.push_back(q.x0);q.x1[1].col(2).setConstant(-1);auto z=t.run(q);assert(z.prefix>0&&z.prefix<1);assert(t.refreshes==2);assert(z.prefix==p.support(q).prefix); // new branch invalidates clearance
 q.x1={q.x0};q.x1[0].col(0)*=-1;assert(!t.run(q).licensed); // folded projection invalidates self proof
 q.x1={q.x0};q.box=2;assert(!t.run(q).licensed);q.box=0;q.complete=false;assert(t.run(q).prefix==0);q.complete=true;q.x0(0,2)=INFINITY;assert(t.run(q).prefix==0);
 q.x0=x;q.x1={x};q.x1[0].col(2).setConstant(-2);auto a=t.run(q),b=control.run(q);assert(a.prefix==b.prefix&&a.licensed==b.licensed);assert(a.prefix<.5);q.x0(0,0)=1e200;assert(t.run(q).prefix==0);
 // Strict endpoint contact with the enclosing ball cannot be reported as closed-step safe.
 X near=x;near.col(0)*=.1;near.col(1)*=.1;near.col(2).setConstant(.5);Panel pn(near,f,e,.25);Family qq{near,{near},0,true};qq.x1[0].col(2).setConstant(.25);Transport tn(pn);assert(tn.run(qq).prefix<1);
 std::cout<<"10 transport regression groups passed\n";
}
