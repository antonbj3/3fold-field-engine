// Preserve the reference block split choices, metadata order and sample bits.
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>
#include <algorithm>
#pragma pack(push,1)
struct Node { int32_t lo[3],shape[3],stored[3],children[2]; int64_t offset; };
#pragma pack(pop)
static_assert(sizeof(Node)==52,"Unexpected node layout");
struct Tree { int lo[3],shape[3],stored[3]; int64_t cost; std::unique_ptr<Tree> left,right; };
struct Packed { std::vector<Node> nodes; std::vector<uint32_t> payload; };
struct Builder {
    const char* data; int64_t stride[3]; int threshold;
    uint32_t read(int x,int y,int z) const {
        uint32_t value; std::memcpy(&value,data+4*(x*stride[0]+y*stride[1]+z),4);return value;
    }
    std::unique_ptr<Tree> build(const int* lo,const int* shape) {
        auto tree=std::make_unique<Tree>();
        int64_t count=1,reduced=1;
        for(int a=0;a<3;++a) { tree->lo[a]=lo[a];tree->shape[a]=shape[a]; count*=shape[a]; }
        for(int axis=0;axis<3;++axis) {
            bool same=true;
            for(int x=0;x<shape[0] && same;++x) for(int y=0;y<shape[1] && same;++y) for(int z=0;z<shape[2] && same;++z) {
                int q[3]={x,y,z},base[3]={x,y,z};base[axis]=0;
                same=read(lo[0]+q[0],lo[1]+q[1],lo[2]+q[2])==read(lo[0]+base[0],lo[1]+base[1],lo[2]+base[2]);
            }
            tree->stored[axis]=same?1:shape[axis];reduced*=tree->stored[axis];
        }
        tree->cost=sizeof(Node)+4*reduced;
        if(reduced<=64 || count<=threshold) return tree;
        int axis=-1;
        for(int a=0;a<3;++a) if(tree->stored[a]>1 && (axis<0 || shape[a]>shape[axis])) axis=a;
        if(axis<0) return tree;
        int ls[3],rs[3],rl[3];
        for(int a=0;a<3;++a) {ls[a]=rs[a]=shape[a];rl[a]=lo[a];}
        ls[axis]=shape[axis]/2;rs[axis]-=ls[axis];rl[axis]+=ls[axis];
        auto left=build(lo,ls),right=build(rl,rs);
        int64_t split=sizeof(Node)+left->cost+right->cost;
        if(split<tree->cost) {tree->cost=split;tree->left=std::move(left);tree->right=std::move(right);}
        return tree;
    }
    int flatten(const Tree& tree,Packed& out) {
        int index=static_cast<int>(out.nodes.size());out.nodes.emplace_back();Node node{};
        for(int a=0;a<3;++a) {node.lo[a]=tree.lo[a];node.shape[a]=tree.shape[a];}
        if(tree.left) {
            node.offset=-1;node.children[0]=flatten(*tree.left,out);node.children[1]=flatten(*tree.right,out);
        } else {
            node.offset=out.payload.size();node.children[0]=node.children[1]=-1;
            for(int a=0;a<3;++a) node.stored[a]=tree.stored[a];
            for(int x=0;x<tree.stored[0];++x) for(int y=0;y<tree.stored[1];++y) for(int z=0;z<tree.stored[2];++z)
                out.payload.push_back(read(tree.lo[0]+x,tree.lo[1]+y,tree.lo[2]+z));
        }
        out.nodes[index]=node;return index;
    }
};
extern "C" void* adaptive_pack(const void* data,const int32_t* shape,int threshold) noexcept {
    try {
        if(!data || !shape || threshold<8 || threshold>4096) return nullptr;
        int64_t total=1;
        for(int a=0;a<3;++a) {if(shape[a]<1 || shape[a]>4096) return nullptr;total*=shape[a];}
        if(total>16777216) return nullptr;
        // Reject non-finite IEEE-754 payloads without arithmetic conversion.
        for(int64_t i=0;i<total;++i) {uint32_t bits;std::memcpy(&bits,static_cast<const char*>(data)+4*i,4);if((bits&0x7f800000u)==0x7f800000u) return nullptr;}
        Builder builder{static_cast<const char*>(data),{static_cast<int64_t>(shape[1])*shape[2],shape[2],1},threshold};
        int lo[3]={0,0,0};auto tree=builder.build(lo,shape);auto packed=std::make_unique<Packed>();
        builder.flatten(*tree,*packed);return packed.release();
    } catch(...) {return nullptr;}
}
extern "C" int64_t adaptive_nodes(const void* pointer) noexcept {return pointer?static_cast<const Packed*>(pointer)->nodes.size():0;}
extern "C" int64_t adaptive_samples(const void* pointer) noexcept {return pointer?static_cast<const Packed*>(pointer)->payload.size():0;}
extern "C" int adaptive_copy(const void* pointer,void* nodes,int64_t count,void* payload,int64_t samples) noexcept {
    if(!pointer || !nodes || !payload) return 1;
    auto& packed=*static_cast<const Packed*>(pointer);
    if(count!=static_cast<int64_t>(packed.nodes.size()) || samples!=static_cast<int64_t>(packed.payload.size())) return 1;
    std::memcpy(nodes,packed.nodes.data(),packed.nodes.size()*sizeof(Node));
    std::memcpy(payload,packed.payload.data(),packed.payload.size()*sizeof(uint32_t));return 0;
}
extern "C" void adaptive_free(void* pointer) noexcept {delete static_cast<Packed*>(pointer);}
