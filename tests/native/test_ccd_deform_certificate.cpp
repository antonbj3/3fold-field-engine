#include "ccd_deform_certificate.hpp"
#include <cassert>
#include <iostream>
using namespace deform_ccd;
int main(){X x(4,3);x<<-1,-1,2,-1,1,2,1,-1,2,1,1,2;F f(2,3);f<<0,2,1,2,3,1;F e(5,2);e<<0,1,0,2,1,2,1,3,2,3;Panel p(x,f,e,.25);Family q{x,{x},0,true};assert(p.injection(q));assert(p.support(q).prefix==1);q.x1[0](0,2)=1;assert(p.support(q).prefix==1);q.x1[0].col(2).setConstant(-2);auto t=p.support(q);assert(t.prefix>0&&t.prefix<.5);q.x1.push_back(x);assert(p.support(q).prefix==t.prefix);q.x1[0]=x;q.box=.001;assert(p.support(q).prefix==1);q.box=1;assert(!p.injection(q));q.box=0;q.x1[0].col(0)*=-1;assert(!p.injection(q));q.complete=false;assert(p.support(q).prefix==0);q.complete=true;q.x1.clear();assert(p.support(q).prefix==0);bool fail=false;try{F bad=f;bad.row(1)=bad.row(0);Panel b(x,bad,e,.25);}catch(...){fail=true;}assert(fail);fail=false;try{F bad=e;bad(0,1)=3;Panel b(x,f,bad,.25);}catch(...){fail=true;}assert(fail);q.x1={x};q.x1[0](0,0)=NAN;assert(!p.injection(q));
 // Interior may hit while all trajectories' endpoints remain outside the ball.
 q.x0=x;q.x0.col(2).setConstant(.5);q.x1={q.x0};q.x1[0].col(2).setConstant(-.5);auto a=p.support(q);assert(a.prefix<.5&&a.prefix>0);
 std::cout<<"12 regression groups passed\n";
}
