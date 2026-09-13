// Separate registration API around the verified immutable fused pipeline.
#include "../mesh_fused_native_v1/field.cu"
extern "C" int fused_pin(void* handle,void* region,size_t bytes) noexcept {
    if(!handle||!region)return 1;auto& f=*static_cast<Fused*>(handle);
    if(static_cast<State*>(f.rt)->owner!=std::this_thread::get_id())return 3;
    if(f.failed)return 2;
    size_t n=f.pair->size,expected=((2*n+63)/64)*64+4*n;
    if(bytes!=expected)return 1;
    return cudaHostRegister(region,bytes,cudaHostRegisterDefault)==cudaSuccess?0:2;
}
extern "C" int fused_unpin(void* handle,void* region) noexcept {
    if(!handle||!region)return 1;auto& f=*static_cast<Fused*>(handle);
    if(static_cast<State*>(f.rt)->owner!=std::this_thread::get_id())return 3;
    return cudaHostUnregister(region)==cudaSuccess?0:2;
}
