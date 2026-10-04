// Optional topology, independent ray parity and AABB broadphase. No field-sign reuse.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>
#include <omp.h>

namespace {
bool faces_ok(const int64_t* t, int64_t nv, int64_t nt) {
    if (!t || nv < 1 || nv >= (int64_t(1)<<31) || nt < 0 || nt > 10000000) return false;
    for (int64_t i=0;i<nt*3;++i) if(t[i]<0 || t[i]>=nv) return false;
    return true;
}
bool vertices_ok(const double* v,int64_t nv) {
    if(!v) return false;
    for(int64_t i=0;i<nv*3;++i) if(!std::isfinite(v[i])) return false;
    return true;
}
uint64_t mix(uint64_t x) { x^=x>>30; x*=0xbf58476d1ce4e5b9ULL; x^=x>>27; x*=0x94d049bb133111ebULL; return x^(x>>31); }
struct Node { double lo[3],hi[3]; int left=-1,right=-1; int64_t begin=0,end=0; };
struct Tree {
    std::vector<double> boxes;
    std::vector<int64_t> ids;
    std::vector<Node> nodes;
    int build(int64_t begin,int64_t end) {
        Node n; n.begin=begin; n.end=end;
        for(int d=0;d<3;++d){n.lo[d]=INFINITY;n.hi[d]=-INFINITY;}
        for(int64_t i=begin;i<end;++i) for(int d=0;d<3;++d){n.lo[d]=std::min(n.lo[d],boxes[6*ids[i]+d]);n.hi[d]=std::max(n.hi[d],boxes[6*ids[i]+3+d]);}
        int index=int(nodes.size()); nodes.push_back(n);
        if(end-begin>16){
            int axis=0; for(int d=1;d<3;++d) if(n.hi[d]-n.lo[d]>n.hi[axis]-n.lo[axis]) axis=d;
            int64_t mid=begin+(end-begin)/2;
            std::nth_element(ids.begin()+begin,ids.begin()+mid,ids.begin()+end,[&](int64_t a,int64_t b){
                double ca=boxes[6*a+axis]*0.5+boxes[6*a+3+axis]*0.5;
                double cb=boxes[6*b+axis]*0.5+boxes[6*b+3+axis]*0.5;
                return ca<cb || (ca==cb && a<b);
            });
            int l=build(begin,mid),r=build(mid,end); nodes[index].left=l;nodes[index].right=r;
        }
        return index;
    }
    static bool intersects(const double* lo,const double* hi,const double* b) {
        for(int d=0;d<3;++d) if(hi[d]<b[d] || lo[d]>b[d+3]) return false;
        return true;
    }
    int64_t query(int index,const double* b,int64_t* out) const {
        const Node& n=nodes[index]; if(!intersects(n.lo,n.hi,b)) return 0;
        if(n.left>=0){int64_t a=query(n.left,b,out);return a+query(n.right,b,out?out+a:nullptr);}
        int64_t count=0;
        for(int64_t i=n.begin;i<n.end;++i){int64_t id=ids[i];if(intersects(&boxes[6*id],&boxes[6*id+3],b)){if(out) out[count]=id;++count;}}
        return count;
    }
};
}
extern "C" {
int meshgate_edges(const int64_t* t,int64_t nv,int64_t nt,int64_t* bad) {
    if(!bad || !faces_ok(t,nv,nt)) return 1;
    try {
        size_t size=1;while(size<size_t(nt)*6+1)size*=2;
        std::vector<uint64_t> keys(size,UINT64_MAX);std::vector<uint32_t> counts(size,0);
        for(int64_t i=0;i<nt;++i) for(int e=0;e<3;++e){
            uint64_t a=uint64_t(t[3*i+e]),b=uint64_t(t[3*i+(e+1)%3]);if(a>b)std::swap(a,b);
            uint64_t key=(a<<32)|b;size_t at=size_t(mix(key))&(size-1);
            while(keys[at]!=UINT64_MAX && keys[at]!=key)at=(at+1)&(size-1);
            keys[at]=key;++counts[at];
        }
        int64_t n=0;for(size_t i=0;i<size;++i)if(counts[i] && counts[i]!=2)++n;
        *bad=n;return 0;
    } catch(...) {return 2;}
}
int meshgate_parity(const double* v,int64_t nv,const int64_t* t,int64_t nt,
                    const double* p,int64_t np,int threads,uint8_t* inside,uint8_t* uncertain) {
    if(!inside || !uncertain || !p || np<0 || np>4096 || threads<1 || threads>4 ||
       !faces_ok(t,nv,nt) || !vertices_ok(v,nv))return 1;
    for(int64_t i=0;i<np*3;++i)if(!std::isfinite(p[i]))return 1;
    struct RayTriangle { double a[3],e1[3],e2[3],pv[3],inv,condition; };
    try {
    std::vector<RayTriangle> rays; rays.reserve(size_t(nt));bool global_unknown=false;
    constexpr double eps=std::numeric_limits<double>::epsilon();
    for(int64_t i=0;i<nt;++i){
        RayTriangle r;
        const double *a=&v[3*t[3*i]],*b=&v[3*t[3*i+1]],*c=&v[3*t[3*i+2]];
        for(int d=0;d<3;++d){r.a[d]=a[d];r.e1[d]=b[d]-a[d];r.e2[d]=c[d]-a[d];}
        r.pv[0]=-r.e2[1];r.pv[1]=r.e2[0];r.pv[2]=0.0;
        double det=(r.e1[0]*r.pv[0]+r.e1[1]*r.pv[1])+r.e1[2]*r.pv[2];
        double de=16*eps*(std::abs(r.e1[0]*r.pv[0])+std::abs(r.e1[1]*r.pv[1])+std::abs(r.e1[2]*r.pv[2]))+1e-300;
        if(!std::isfinite(det)){global_unknown=true;continue;}
        if(std::abs(std::abs(det)-1e-9)<=de)global_unknown=true;
        if(std::abs(det)<=1e-9)continue;
        r.inv=1.0/det;r.condition=de*std::abs(r.inv);
        if(r.condition>0.125)global_unknown=true;
        rays.push_back(r);
    }
    #pragma omp parallel for num_threads(threads)
    for(int64_t j=0;j<np;++j){
        int64_t hits=0;bool unknown=global_unknown;
        for(const RayTriangle& r:rays){
            const double *e1=r.e1,*e2=r.e2,*pv=r.pv;
            double tv[3],q[3];for(int d=0;d<3;++d)tv[d]=p[3*j+d]-r.a[d];
            double inv=r.inv,det_condition=r.condition;
            double u=((tv[0]*pv[0]+tv[1]*pv[1])+tv[2]*pv[2])*inv;
            double ue=64*eps*((std::abs(tv[0]*pv[0])+std::abs(tv[1]*pv[1])+std::abs(tv[2]*pv[2]))*std::abs(inv)+std::abs(u))+4*std::abs(u)*det_condition+1e-300;
            if(!std::isfinite(u)||!std::isfinite(ue)){unknown=true;continue;}
            if(std::abs(u+1e-7)<=ue || std::abs(u-(1+1e-7))<=ue)unknown=true;
            if(u < -1e-7 || u>1+1e-7)continue;
            q[0]=tv[1]*e1[2]-tv[2]*e1[1];q[1]=tv[2]*e1[0]-tv[0]*e1[2];q[2]=tv[0]*e1[1]-tv[1]*e1[0];
            double vv=((0.0*q[0]+0.0*q[1])+q[2])*inv;
            double ve=64*eps*(std::abs(q[2]*inv)+std::abs(vv))+4*std::abs(vv)*det_condition+1e-300;
            double uv=u+vv,uve=ue+ve+4*eps*(std::abs(u)+std::abs(vv));
            if(!std::isfinite(vv)||!std::isfinite(uve)){unknown=true;continue;}
            if(std::abs(vv+1e-7)<=ve || std::abs(uv-(1+1e-7))<=uve)unknown=true;
            if(vv < -1e-7 || uv>1+1e-7)continue;
            double z=((e2[0]*q[0]+e2[1]*q[1])+e2[2]*q[2])*inv;
            double ze=64*eps*((std::abs(e2[0]*q[0])+std::abs(e2[1]*q[1])+std::abs(e2[2]*q[2]))*std::abs(inv)+std::abs(z))+4*std::abs(z)*det_condition+1e-300;
            if(!std::isfinite(z)||!std::isfinite(ze)){unknown=true;continue;}
            if(std::abs(z-1e-7)<=ze)unknown=true;
            if(z>1e-7)++hits;
        }
        inside[j]=uint8_t(hits%2);uncertain[j]=uint8_t(unknown);
    }
    } catch(...) {return 2;}
    return 0;
}
void* meshgate_tree_new(const double* v,int64_t nv,const int64_t* t,int64_t nt) {
    if(nt<1 || !faces_ok(t,nv,nt) || !vertices_ok(v,nv))return nullptr;
    try {
        Tree tree;tree.boxes.resize(size_t(nt)*6);tree.ids.resize(size_t(nt));std::iota(tree.ids.begin(),tree.ids.end(),0);
        for(int64_t i=0;i<nt;++i)for(int d=0;d<3;++d){double a=v[3*t[3*i]+d],b=v[3*t[3*i+1]+d],c=v[3*t[3*i+2]+d];tree.boxes[6*i+d]=std::min(a,std::min(b,c));tree.boxes[6*i+3+d]=std::max(a,std::max(b,c));}
        tree.nodes.reserve(size_t(nt)/4+1);tree.build(0,nt);return new Tree(std::move(tree));
    }catch(...){return nullptr;}
}
void meshgate_tree_free(void* handle) {delete static_cast<Tree*>(handle);}
int meshgate_tree_query(void* handle,const double* boxes,int64_t np,int64_t* offsets,int64_t* ids,int64_t capacity) {
    if(!handle || !boxes || !offsets || np<0 || np>4096 || capacity<0)return 1;
    for(int64_t i=0;i<np;++i)for(int d=0;d<3;++d)if(!std::isfinite(boxes[6*i+d]) || !std::isfinite(boxes[6*i+3+d]) || boxes[6*i+d]>boxes[6*i+3+d])return 1;
    try{
        const Tree& tree=*static_cast<Tree*>(handle);std::vector<int64_t> off(size_t(np)+1,0);
        for(int64_t i=0;i<np;++i)off[size_t(i)+1]=off[size_t(i)]+tree.query(0,&boxes[6*i],nullptr);
        if(ids && off.back()>capacity)return 2;
        if(ids)for(int64_t i=0;i<np;++i)tree.query(0,&boxes[6*i],ids+off[size_t(i)]);
        std::copy(off.begin(),off.end(),offsets);return 0;
    }catch(...){return 3;}
}
}
