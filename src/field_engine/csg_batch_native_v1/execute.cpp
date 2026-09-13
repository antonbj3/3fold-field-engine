// Batched tile execution; NumPy-compatible ordered float32 min/max/offset.
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
extern "C" int csg_gather(const void*,int64_t,const void*,int64_t,const int32_t*,const int32_t*,void*) noexcept;
struct Instruction {int32_t op,left,right,pad;double offset;};
static_assert(sizeof(Instruction)==24,"Unexpected instruction layout");
extern "C" int csg_execute(const void* const* nodes,const int64_t* counts,const void* const* payload,
                            const int64_t* lengths,int32_t fields,const int32_t* shape,
                            const Instruction* program,int32_t instructions,float* output) noexcept {
    try {
        if(!nodes||!counts||!payload||!lengths||!shape||!program||!output||fields<1||fields>256||instructions<1||instructions>1024) return 1;
        int64_t total=1;for(int a=0;a<3;++a){if(shape[a]<1||shape[a]>4096)return 1;total*=shape[a];}if(total>16777216)return 1;
        std::vector<int> last(fields+instructions,-1);
        for(int n=0;n<instructions;++n){
            const auto& p=program[n];
            if(p.op<0||p.op>2||p.left<0||p.left>=fields+n)return 2;
            last[p.left]=n;
            if(p.op==2){if(!std::isfinite(p.offset))return 2;}
            else {if(p.right<0||p.right>=fields+n)return 2;last[p.right]=n;}
        }
        last.back()=instructions;
        for(int x=0;x<shape[0];x+=16)for(int y=0;y<shape[1];y+=32)for(int z=0;z<shape[2];z+=16){
            int32_t lo[3]={x,y,z},hi[3]={std::min(x+16,shape[0]),std::min(y+32,shape[1]),std::min(z+16,shape[2])};
            int size=(hi[0]-x)*(hi[1]-y)*(hi[2]-z);
            std::vector<std::vector<float>> values(fields+instructions);
            for(int f=0;f<fields;++f)if(last[f]>=0){values[f].resize(size);if(csg_gather(nodes[f],counts[f],payload[f],lengths[f],lo,hi,values[f].data()))return 3;}
            for(int n=0;n<instructions;++n){
                const auto& p=program[n];auto& result=values[fields+n];result.resize(size);
                const auto& a=values[p.left];
                for(int i=0;i<size;++i){
                    if(p.op==2) result[i]=static_cast<float>(static_cast<double>(a[i])-p.offset);
                    else {
                        float b=p.op==0?values[p.right][i]:-values[p.right][i];
                        // NumPy chooses the second operand on equal values, including signed zeros.
                        result[i]=p.op==0?(a[i]<b?a[i]:b):(a[i]>b?a[i]:b);
                    }
                    if(!std::isfinite(result[i]))return 4;
                }
                if(last[p.left]==n)std::vector<float>().swap(values[p.left]);
                if(p.op!=2 && p.right!=p.left && last[p.right]==n)std::vector<float>().swap(values[p.right]);
                if(last[fields+n]<0)std::vector<float>().swap(result);
            }
            const auto& result=values.back();int local=0;
            for(int a=x;a<hi[0];++a)for(int b=y;b<hi[1];++b){
                int length=hi[2]-z;int64_t index=(static_cast<int64_t>(a)*shape[1]+b)*shape[2]+z;
                std::memcpy(output+index,result.data()+local,length*sizeof(float));local+=length;
            }
        }
        return 0;
    }catch(...){return 5;}
}
