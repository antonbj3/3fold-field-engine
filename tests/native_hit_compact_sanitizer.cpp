// ASan/UBSan checks for equal-key stability, signed-zero bits and invalid counts.
#include <cstdint>
#include <cstring>
#include <limits>
#include <iostream>
#include <vector>
extern "C" int compact_two_hits(const int32_t*,const double*,const int32_t*,int64_t,int,int64_t*,double*,int32_t*,int64_t*) noexcept;
int main(){
    int32_t counts[]={2,1,0,2,2};
    double nan=std::numeric_limits<double>::quiet_NaN();
    double depths[]={2.,1.,-0.,nan,nan,nan,0.,-0.,-0.,0.};
    int32_t signs[]={1,-1,1,7,7,7,1,-1,-1,-1};
    int64_t columns[10],n=-1;double z[10];int32_t s[10];
    if(compact_two_hits(counts,depths,signs,5,2,columns,z,s,&n) || n!=7)return 1;
    const int64_t ec[]={0,0,1,3,3,4,4};
    const double ez[]={1.,2.,-0.,-0.,0.,-0.,0.};
    const int32_t es[]={-1,1,1,-1,1,-1,-1};
    if(std::memcmp(columns,ec,sizeof(ec)) || std::memcmp(z,ez,sizeof(ez)) || std::memcmp(s,es,sizeof(es)))return 2;
    int cases=1;
    for(int bad:{-1,3}){
        counts[4]=bad;n=123;double saved[10];std::memcpy(saved,z,sizeof(z));
        if(compact_two_hits(counts,depths,signs,5,2,columns,z,s,&n)!=1 || n!=123 || std::memcmp(saved,z,sizeof(z)))return 3;
        ++cases;
    }
    counts[4]=2;
    if(compact_two_hits(counts,depths,signs,0,0,columns,z,s,&n) || n!=0)return 4;++cases;
    for(int width:{-1,3}){
        if(compact_two_hits(counts,depths,signs,5,width,columns,z,s,&n)!=1)return 5;++cases;
    }
    for(int64_t rows:{int64_t(-1),int64_t(262145)}){
        if(compact_two_hits(counts,depths,signs,rows,2,columns,z,s,&n)!=1)return 6;++cases;
    }
    std::vector<int32_t> zeros(262144,0),orientations(524288,1),outs(524288);
    std::vector<double> empty(524288,nan),outz(524288);
    std::vector<int64_t> outc(524288);
    for(int width:{0,1,2}){
        if(compact_two_hits(zeros.data(),empty.data(),orientations.data(),zeros.size(),width,outc.data(),outz.data(),outs.data(),&n) || n!=0)return 7;++cases;
    }
    std::cout<<"{\"cases\":"<<cases<<",\"exact\":true,\"sanitizers\":\"address,undefined\"}\n";
}
