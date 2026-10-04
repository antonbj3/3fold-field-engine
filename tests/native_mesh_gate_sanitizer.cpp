#include "../src/field_engine/mesh_gate_native_v1/gate.cpp"
#include <cassert>
#include <iostream>
#include <set>
int main(){
    const double v[]={0,0,0, 4,0,0, 4,4,0, 0,4,0, 0,0,4, 4,0,4, 4,4,4, 0,4,4};
    const int64_t f[]={0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,2,3,7,2,7,6,1,2,6,1,6,5,3,0,4,3,4,7};
    int cases=0,controls=0;
    int64_t guard[]={731,-99,732};
    assert(meshgate_edges(f,8,12,guard+1)==0 && guard[1]==0);++cases;
    assert(guard[0]==731&&guard[2]==732);
    assert(meshgate_edges(f,8,11,guard+1)==0&&guard[1]==3);++cases;
    std::vector<int64_t> dup(f,f+36);dup.insert(dup.end(),f,f+36);
    assert(meshgate_edges(dup.data(),8,24,guard+1)==0&&guard[1]==18);++cases;
    const int64_t repeated[]={0,0,0,0,0,0};
    assert(meshgate_edges(repeated,8,2,guard+1)==0&&guard[1]==1);++cases;
    assert(meshgate_edges(f,8,0,guard+1)==0&&guard[1]==0);++cases;
    auto badedge=[&](const int64_t* t,int64_t nv,int64_t nt,int64_t* out){guard[1]=-99;assert(meshgate_edges(t,nv,nt,out)!=0);assert(guard[0]==731&&guard[1]==-99&&guard[2]==732);++controls;};
    std::vector<int64_t> bad(f,f+36);bad[0]=-1;badedge(bad.data(),8,12,guard+1);bad[0]=8;badedge(bad.data(),8,12,guard+1);
    badedge(nullptr,8,12,guard+1);badedge(f,0,12,guard+1);badedge(f,8,-1,guard+1);badedge(f,8,10000001,guard+1);badedge(f,int64_t(1)<<31,12,guard+1);badedge(f,8,12,nullptr);
    const int64_t tri[]={0,1,2};const double p[]={1,1,-1,8,8,-1,1,1,1};
    uint8_t out[]={71,99,99,99,72},unc[]={81,99,99,99,82};
    assert(meshgate_parity(v,8,tri,1,p,3,1,out+1,unc+1)==0);
    assert(out[1]==1&&out[2]==0&&out[3]==0&&out[0]==71&&out[4]==72&&unc[0]==81&&unc[4]==82);++cases;
    uint8_t out4[3],unc4[3];assert(meshgate_parity(v,8,tri,1,p,3,4,out4,unc4)==0);
    for(int i=0;i<3;++i)assert(out4[i]==out[i+1]&&unc4[i]==unc[i+1]);++cases;
    auto badpar=[&](const double* vv,const int64_t* ff,const double* pp,int64_t np,int threads,uint8_t* oo,uint8_t* uu){std::fill(out+1,out+4,99);assert(meshgate_parity(vv,8,ff,1,pp,np,threads,oo,uu)!=0);for(int i=1;i<4;++i)assert(out[i]==99);assert(out[0]==71&&out[4]==72);++controls;};
    badpar(nullptr,tri,p,3,4,out+1,unc+1);badpar(v,nullptr,p,3,4,out+1,unc+1);badpar(v,tri,nullptr,3,4,out+1,unc+1);
    badpar(v,tri,p,-1,4,out+1,unc+1);badpar(v,tri,p,4097,4,out+1,unc+1);badpar(v,tri,p,3,0,out+1,unc+1);badpar(v,tri,p,3,5,out+1,unc+1);
    badpar(v,tri,p,3,4,nullptr,unc+1);badpar(v,tri,p,3,4,out+1,nullptr);
    double nanv[24];std::copy(v,v+24,nanv);nanv[2]=NAN;badpar(nanv,tri,p,3,4,out+1,unc+1);
    double nanp[9];std::copy(p,p+9,nanp);nanp[1]=INFINITY;badpar(v,tri,nanp,3,4,out+1,unc+1);
    badpar(v,bad.data(),p,3,4,out+1,unc+1);
    void* tree=meshgate_tree_new(v,8,f,12);assert(tree);++cases;
    const double boxes[]={-1,-1,-1,5,5,5, 9,9,9,10,10,10, 1,1,3,2,2,5};
    int64_t off[]={73,-9,-9,-9,-9,74};
    assert(meshgate_tree_query(tree,boxes,3,off+1,nullptr,0)==0&&off[1]==0&&off[2]==12&&off[3]==12&&off[4]==14&&off[0]==73&&off[5]==74);++cases;
    int64_t ids[16];std::fill(ids,ids+16,-9);ids[0]=83;ids[15]=84;
    assert(meshgate_tree_query(tree,boxes,3,off+1,ids+1,14)==0);
    std::set<int64_t> all(ids+1,ids+13),top(ids+13,ids+15);assert(all.size()==12&&top==std::set<int64_t>({2,3})&&ids[0]==83&&ids[15]==84);++cases;
    std::fill(off+1,off+5,-9);std::fill(ids+1,ids+15,-9);
    assert(meshgate_tree_query(tree,boxes,3,off+1,ids+1,1)==2);for(int i=1;i<5;++i)assert(off[i]==-9);for(int i=1;i<15;++i)assert(ids[i]==-9);++controls;
    assert(meshgate_tree_query(nullptr,boxes,3,off+1,nullptr,0)!=0);++controls;
    assert(meshgate_tree_query(tree,nullptr,3,off+1,nullptr,0)!=0);++controls;
    assert(meshgate_tree_query(tree,boxes,-1,off+1,nullptr,0)!=0);++controls;
    assert(meshgate_tree_query(tree,boxes,4097,off+1,nullptr,0)!=0);++controls;
    assert(meshgate_tree_query(tree,boxes,3,nullptr,nullptr,0)!=0);++controls;
    assert(meshgate_tree_query(tree,boxes,3,off+1,nullptr,-1)!=0);++controls;
    double bb[6]={NAN,0,0,1,1,1};assert(meshgate_tree_query(tree,bb,1,off+1,nullptr,0)!=0);++controls;
    bb[0]=2;assert(meshgate_tree_query(tree,bb,1,off+1,nullptr,0)!=0);++controls;
    assert(off[0]==73&&off[5]==74&&ids[0]==83&&ids[15]==84);
    meshgate_tree_free(tree);meshgate_tree_free(nullptr);
    assert(!meshgate_tree_new(v,8,bad.data(),12));++controls;
    assert(!meshgate_tree_new(nanv,8,f,12));++controls;
    assert(!meshgate_tree_new(v,8,f,0));++controls;
    assert(!meshgate_tree_new(nullptr,8,f,12));++controls;
    // Force internal BVH nodes and compare every returned face against brute boxes.
    std::vector<int64_t> many;for(int i=0;i<20;++i)many.insert(many.end(),f,f+36);
    tree=meshgate_tree_new(v,8,many.data(),240);assert(tree);
    int64_t off2[4];assert(meshgate_tree_query(tree,boxes,3,off2,nullptr,0)==0);
    std::vector<int64_t> ids2(static_cast<size_t>(off2[3]), -1);assert(meshgate_tree_query(tree,boxes,3,off2,ids2.data(),off2[3])==0);
    for(int q=0;q<3;++q){std::set<int64_t> expected,actual(ids2.begin()+off2[q],ids2.begin()+off2[q+1]);
        for(int64_t i=0;i<240;++i){double lo[3],hi[3];for(int d=0;d<3;++d){lo[d]=hi[d]=v[3*many[3*i]+d];for(int e=1;e<3;++e){lo[d]=std::min(lo[d],v[3*many[3*i+e]+d]);hi[d]=std::max(hi[d],v[3*many[3*i+e]+d]);}}
            if(Tree::intersects(lo,hi,boxes+6*q))expected.insert(i);}
        assert(expected==actual);}
    meshgate_tree_free(tree);++cases;
    std::cout<<"{\"cases\":"<<cases<<",\"controls\":"<<controls<<",\"canaries\":true,\"exact\":true}\n";
}
