// Exact adaptive tile reads with contiguous last-axis row copies.
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <vector>
#pragma pack(push, 1)
struct Node { int32_t lo[3], shape[3], stored[3], children[2]; int64_t offset; };
#pragma pack(pop)
static_assert(sizeof(Node)==52, "Unexpected packed node layout");

extern "C" int csg_gather(const void* raw, int64_t count, const void* payload,
                           int64_t samples, const int32_t* lower, const int32_t* upper,
                           void* output) noexcept {
    try {
        if (!raw || !payload || !lower || !upper || !output || count<1 || count>33554432 || samples<1 || samples>16777216) return 1;
        int64_t total=1, dims[3];
        for(int a=0;a<3;++a) {
            if(lower[a]<0 || upper[a]<=lower[a] || upper[a]>4096) return 2;
            dims[a]=upper[a]-lower[a]; total*=dims[a];
        }
        if(total>8192) return 2;
        std::vector<uint8_t> coverage(static_cast<size_t>(total),0);
        for(int64_t n=0;n<count;++n) {
            Node node; std::memcpy(&node,static_cast<const char*>(raw)+n*sizeof(Node),sizeof(Node));
            if(node.children[0]>=0) continue;
            int lo[3],hi[3]; bool overlap=true; int64_t size=1;
            for(int a=0;a<3;++a) {
                if(node.lo[a]<0 || node.shape[a]<1 || node.shape[a]>4096 || node.lo[a]>4096-node.shape[a] ||
                   (node.stored[a]!=1 && node.stored[a]!=node.shape[a])) return 3;
                size*=node.stored[a];
                lo[a]=std::max(lower[a],node.lo[a]); hi[a]=std::min(upper[a],node.lo[a]+node.shape[a]);
                overlap &= hi[a]>lo[a];
            }
            if(node.offset<0 || size>samples || node.offset>samples-size) return 3;
            if(!overlap) continue;
            // A leaf's last axis is either contiguous or a single repeated sample.
            const int64_t width=hi[2]-lo[2];
            for(int x=lo[0];x<hi[0];++x) for(int y=lo[1];y<hi[1];++y) {
                int64_t source=node.offset;
                source+=(node.stored[0]==1?0:x-node.lo[0])*static_cast<int64_t>(node.stored[1])*node.stored[2];
                source+=(node.stored[1]==1?0:y-node.lo[1])*static_cast<int64_t>(node.stored[2]);
                source+=node.stored[2]==1?0:lo[2]-node.lo[2];
                const int64_t target=((x-lower[0])*dims[1]+y-lower[1])*dims[2]+lo[2]-lower[2];
                if(std::memchr(coverage.data()+target,1,static_cast<size_t>(width))) return 4;
                std::memset(coverage.data()+target,1,static_cast<size_t>(width));
                char* dest=static_cast<char*>(output)+4*target;
                const char* src=static_cast<const char*>(payload)+4*source;
                if(node.stored[2]!=1) {
                    std::memcpy(dest,src,static_cast<size_t>(4*width));
                } else {
                    for(int64_t z=0;z<width;++z) std::memcpy(dest+4*z,src,4);
                }
            }
        }
        for(auto flag:coverage) if(!flag) return 4;
        return 0;
    } catch(...) { return 5; }
}
