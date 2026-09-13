// Stable two-hit compare/swap and bit-preserving row compaction.
#include <cstdint>
#include <cstring>
extern "C" int compact_two_hits(const int32_t* counts,const double* depths,
    const int32_t* signs,int64_t rows,int width,int64_t* columns,
    double* out_depths,int32_t* out_signs,int64_t* written) noexcept {
    if(!counts || !depths || !signs || !columns || !out_depths || !out_signs || !written ||
       rows<0 || rows>262144 || width<0 || width>2)return 1;
    // Validate all counts before writing any output; allocated capacity is rows*width.
    for(int64_t row=0;row<rows;++row)if(counts[row]<0 || counts[row]>width)return 1;
    int64_t n=0;
    for(int64_t row=0;row<rows;++row){
        const int count=counts[row];if(!count)continue;
        const int64_t base=row*width;
        const bool swap=count==2 && (depths[base]>depths[base+1] ||
            (depths[base]==depths[base+1] && signs[base]>signs[base+1]));
        for(int item=0;item<count;++item){
            const int64_t source=base+(swap?1-item:item);
            columns[n]=row;
            std::memcpy(out_depths+n,depths+source,sizeof(double));
            out_signs[n]=signs[source];++n;
        }
    }
    *written=n;return 0;
}
