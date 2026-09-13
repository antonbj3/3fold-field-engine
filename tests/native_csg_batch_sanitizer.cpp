// Standalone ASan/UBSan exercise including ordered batch evaluation.
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>
extern "C" void* adaptive_pack(const void*,const int32_t*,int) noexcept;
extern "C" int64_t adaptive_nodes(const void*) noexcept;
extern "C" int64_t adaptive_samples(const void*) noexcept;
extern "C" int adaptive_copy(const void*,void*,int64_t,void*,int64_t) noexcept;
extern "C" void adaptive_free(void*) noexcept;
extern "C" int csg_gather(const void*,int64_t,const void*,int64_t,const int32_t*,const int32_t*,void*) noexcept;
struct Instruction {int32_t op,left,right,pad;double offset;};
extern "C" int csg_execute(const void* const*,const int64_t*,const void* const*,const int64_t*,int32_t,const int32_t*,const Instruction*,int32_t,float*) noexcept;
int main() {
    int cases=0;
    for(int mode=0;mode<8;++mode) for(int threshold:{8,512,4096}) {
        int32_t shape[3]={19,21,17}; if(mode<3) shape[mode]=1;
        int64_t total=static_cast<int64_t>(shape[0])*shape[1]*shape[2];
        std::vector<float> input(total);
        for(int64_t i=0;i<total;++i) input[i]=mode==3?0.25f:static_cast<float>((i*37)%101-50)*0.03125f;
        if(mode==4) for(int64_t i=0;i<total;i+=2) input[i]=-0.0f;
        if(mode==5) for(int64_t i=0;i<total;++i) input[i]=static_cast<float>(i%shape[2]);
        void* packed=adaptive_pack(input.data(),shape,threshold);if(!packed)return 1;
        int64_t nn=adaptive_nodes(packed),ns=adaptive_samples(packed);
        std::vector<char> nodes(nn*52);std::vector<float> samples(ns);
        if(adaptive_copy(packed,nodes.data(),nn,samples.data(),ns))return 2;
        adaptive_free(packed);
        for(int x=0;x<shape[0];x+=7)for(int y=0;y<shape[1];y+=7)for(int z=0;z<shape[2];z+=7){
            int32_t lo[3]={x,y,z},hi[3]={std::min(x+7,shape[0]),std::min(y+7,shape[1]),std::min(z+7,shape[2])};
            int64_t n=static_cast<int64_t>(hi[0]-x)*(hi[1]-y)*(hi[2]-z);std::vector<float> tile(n);
            if(csg_gather(nodes.data(),nn,samples.data(),ns,lo,hi,tile.data()))return 3;
            int64_t q=0;
            for(int a=x;a<hi[0];++a)for(int b=y;b<hi[1];++b)for(int c=z;c<hi[2];++c){
                int64_t i=(static_cast<int64_t>(a)*shape[1]+b)*shape[2]+c;
                if(std::memcmp(&tile[q++],&input[i],4))return 4;
            }
        }
        const void* node_ptrs[2]={nodes.data(),nodes.data()};
        const void* sample_ptrs[2]={samples.data(),samples.data()};
        int64_t counts[2]={nn,nn},lengths[2]={ns,ns};
        Instruction program[4]={{0,0,1,0,0},{2,2,0,0,.1},{2,3,0,0,-.1},{1,4,0,0,0}};
        std::vector<float> result(total);
        if(csg_execute(node_ptrs,counts,sample_ptrs,lengths,2,shape,program,4,result.data()))return 5;
        for(int64_t i=0;i<total;++i){
            float a=static_cast<float>(static_cast<double>(input[i])-.1);
            a=static_cast<float>(static_cast<double>(a)+.1);
            float b=-input[i],expected=a>b?a:b;
            if(std::memcmp(&expected,&result[i],4))return 6;
        }
        Instruction bad{0,0,99,0,0};
        if(csg_execute(node_ptrs,counts,sample_ptrs,lengths,2,shape,&bad,1,result.data())!=2)return 7;
        Instruction overflow{2,0,0,0,1e39};
        if(csg_execute(node_ptrs,counts,sample_ptrs,lengths,2,shape,&overflow,1,result.data())!=4)return 8;
        ++cases;
    }
    std::cout << "{\"cases\":" << cases << ",\"exact\":true,\"sanitizers\":\"address,undefined\"}\n";
}
