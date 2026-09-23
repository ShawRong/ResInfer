// SPDX-License-Identifier: Apache-2.0
// SM100 MXFP4: NVFP4 pack16/QDQ/mask/warp dataflow, block32 E8M0 ceil.
// FP32 source and residual stay FP32. No global alpha or shared gather.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include "fp4/nvfp4_utils.cuh"

namespace {
using namespace vllm;
struct Scale { uint8_t code; float inverse, value; };
__device__ __forceinline__ Scale make_scale(float amax) {
  auto c=__nv_cvt_float_to_e8m0(__fmul_rn(amax,reciprocal_approximate_ftz(6.f)),
                              __NV_SATFINITE,cudaRoundPosInf);
  return {c,c==0?0.f:__uint_as_float(uint32_t(254-c)<<23),
          __uint_as_float(c==0?0x00400000u:uint32_t(c)<<23)};
}
template<int WIDTH>
__device__ __forceinline__ float reduce_max(float v) {
  #pragma unroll
  for(int d=1;d<WIDTH;d*=2) v=fmaxf(v,__shfl_xor_sync(0xffffffffu,v,d));
  return v;
}
__device__ __forceinline__ int64_t sf_offset(int row,int g,int groups,bool swizzle) {
  if(!swizzle) return int64_t(row)*groups+g;
  return (int64_t(row/128)*((groups+3)/4)+g/4)*512+(row%32)*16+(row%128/32)*4+g%4;
}

template<class T>
__device__ __forceinline__ void read16(T const* p,int64_t stride,float (&v)[16]) {
  if constexpr(!std::is_same_v<T,float>) {
    if(stride==1 && (reinterpret_cast<uintptr_t>(p)&31)==0) {
      PackedVec<T> lo,hi;
      load_packed_vec16(p,lo,hi); // production NVFP4 256-bit load
      #pragma unroll
      for(int j=0;j<4;++j) {
        float2 a,b;
        if constexpr(std::is_same_v<T,half>) {
          a=__half22float2(lo.elts[j]);b=__half22float2(hi.elts[j]);
        } else {
          a=__bfloat1622float2(lo.elts[j]);b=__bfloat1622float2(hi.elts[j]);
        }
        v[2*j]=a.x;v[2*j+1]=a.y;v[8+2*j]=b.x;v[9+2*j]=b.y;
      }
      return;
    }
  } else if(stride==1 && (reinterpret_cast<uintptr_t>(p)&15)==0) {
    #pragma unroll
    for(int j=0;j<4;++j) {
      float4 a=reinterpret_cast<float4 const*>(p)[j];
      v[4*j]=a.x;v[4*j+1]=a.y;v[4*j+2]=a.z;v[4*j+3]=a.w;
    }
    return;
  }
  #pragma unroll
  for(int j=0;j<16;++j) v[j]=float(p[j*stride]);
}

template<bool QDQ>
__device__ __forceinline__ uint2 encode16(float (&v)[16],Scale s) {
  uint2 q;
  #pragma unroll
  for(int pack=0;pack<2;++pack) {
    float2 a[4];
    #pragma unroll
    for(int j=0;j<4;++j) a[j]=make_float2(__fmul_rn(v[8*pack+2*j],s.inverse),
                                                       __fmul_rn(v[8*pack+2*j+1],s.inverse));
    uint32_t bits;
    if constexpr(QDQ) bits=fp32x8_to_e2m1_with_dequant(a);
    else bits=fp32_vec_to_e2m1(a);
    if(pack==0) q.x=bits;else q.y=bits;
    if constexpr(QDQ) {
      #pragma unroll
      for(int j=0;j<4;++j) {
        v[8*pack+2*j]=__fmul_rn(a[j].x,s.value);
        v[8*pack+2*j+1]=__fmul_rn(a[j].y,s.value);
      }
    }
  }
  return q;
}

// R=0: standard, R=8: full-R1 row pairs, R=1/2/4: fixed selection per 8.
template<class T,int R>
__global__ void fused_quant(T const* x,int64_t s0,int64_t s1,
    uint8_t const* masks,uint8_t* q,uint8_t* sf,
    int m,int k,int kp,int output_rows,int padded_rows,bool swizzle,bool tile8,int pdl_mode) {
  // Trigger every CTA, including padding CTAs. The consumer must wait before
  // reading our outputs; this permits only its independent setup to overlap.
  if(pdl_mode==2 && threadIdx.x==0) cudaTriggerProgrammaticLaunchCompletion();
  constexpr bool SELECTIVE=R>0 && R<8;
  constexpr int FACTOR=R==8?2:1;
  int row=blockIdx.x;
  int outrow=(R==8 && tile8)?(row/8)*16+row%8:row*FACTOR;
  int resrow=(R==8 && tile8)?outrow+8:outrow+1;
  int groups=kp/32,selected=SELECTIVE?k/8*R:0;
  bool valid=row<m;
  int written_groups=k/32+(SELECTIVE?(selected+31)/32:0);
  int sf_groups=swizzle?(groups+3)/4*4:groups;
  // Clear only padding/invalid rows, never overwrite valid producer stores.
  for(int g=threadIdx.x;g<sf_groups;g+=blockDim.x) {
    if((!valid || g>=written_groups) && outrow<(swizzle?padded_rows:output_rows)) {
      sf[sf_offset(outrow,g,groups,swizzle)]=0;
      if constexpr(FACTOR==2) sf[sf_offset(resrow,g,groups,swizzle)]=0;
    }
  }
  if(row>=m) {
    // A4/A6 read a padded tile8 activation as real GEMM rows. Zero its data
    // as well as the scales; the old row-pair path has no physical padding.
    if constexpr(R==8) if(tile8 && row<output_rows/2)
      for(int j=threadIdx.x;j<kp/2;j+=blockDim.x) {
        q[int64_t(outrow)*(kp/2)+j]=0;
        q[int64_t(resrow)*(kp/2)+j]=0;
      }
    if(pdl_mode==1) {
      __syncthreads();
      if(threadIdx.x==0) cudaTriggerProgrammaticLaunchCompletion();
    }
    return;
  }
  for(int tile=0;tile<(k+blockDim.x*16-1)/(blockDim.x*16);++tile) {
    int g=tile*blockDim.x+threadIdx.x;
    bool has_input=g<k/16;
    float original[16]={},v[16];
    if(has_input) {
      T const* p=x+int64_t(row)*s0+int64_t(g*16)*s1;
      read16(p,s1,original);
    }
    float amax=0.f;
    #pragma unroll
    for(int j=0;j<16;++j) {v[j]=original[j];amax=fmaxf(amax,fabsf(v[j]));}
    Scale base=make_scale(reduce_max<2>(amax));
    uint2 bits=encode16<(R>0)>(v,base);
    if(has_input) {
      *reinterpret_cast<uint2*>(q+int64_t(outrow)*(kp/2)+g*8)=bits;
      if((g&1)==0) sf[sf_offset(outrow,g/2,groups,swizzle)]=base.code;
    }
    if constexpr(R==8) {
      float mx=0.f;
      #pragma unroll
      for(int j=0;j<16;++j) {v[j]=__fsub_rn(original[j],v[j]);mx=fmaxf(mx,fabsf(v[j]));}
      Scale res=make_scale(reduce_max<2>(mx));
      uint2 rbits=encode16<false>(v,res);
      if(has_input) {
        *reinterpret_cast<uint2*>(q+int64_t(resrow)*(kp/2)+g*8)=rbits;
        if((g&1)==0) sf[sf_offset(resrow,g/2,groups,swizzle)]=res.code;
      }
    } else if constexpr(SELECTIVE) {
      constexpr int NR=2*R, SF_THREADS=32/NR, PACK_THREADS=8/NR;
      float res[NR]={};
      #pragma unroll
      for(int half=0;half<2;++half) {
        unsigned mask=has_input?masks[g*2+half]:0;
        #pragma unroll
        for(int j=0;j<R;++j) {
          int index=__ffs(mask)-1;
          float value=0.f;
          // Compile-time register selects: never dynamically index a local array.
          #pragma unroll
          for(int c=0;c<8;++c)
            if(c==index) value=__fsub_rn(original[half*8+c],v[half*8+c]);
          res[half*R+j]=value;mask&=mask-1;
        }
      }
      float mx=0.f;
      #pragma unroll
      for(int j=0;j<NR;++j) mx=fmaxf(mx,fabsf(res[j]));
      Scale rs=make_scale(reduce_max<SF_THREADS>(mx));
      int output_start=g*NR;
      if(threadIdx.x%SF_THREADS==0 && output_start<selected)
        sf[sf_offset(row,k/32+output_start/32,groups,swizzle)]=rs.code;
      float values[8];
      int leader=(threadIdx.x%32)/PACK_THREADS*PACK_THREADS;
      #pragma unroll
      for(int j=0;j<8;++j)
        values[j]=__fmul_rn(__shfl_sync(0xffffffffu,res[j%NR],leader+j/NR),rs.inverse);
      uint32_t packed=fp32_vec_to_e2m1(values);
      if(threadIdx.x%PACK_THREADS==0 && output_start<selected)
        *reinterpret_cast<uint32_t*>(q+int64_t(row)*(kp/2)+k/2+output_start/2)=packed;
    }
  }
  if constexpr(SELECTIVE)
    for(int j=k/2+((selected+7)/8)*4+threadIdx.x;j<kp/2;j+=blockDim.x)
      q[int64_t(row)*(kp/2)+j]=0;
  if(pdl_mode==1) {
    __syncthreads();
    if(threadIdx.x==0) cudaTriggerProgrammaticLaunchCompletion();
  }
}

template<class T>
void dispatch(torch::Tensor x,torch::Tensor q,torch::Tensor sf,
    uint8_t const* masks,int per8,bool residue,bool swizzle,bool tile8,int pdl_mode,cudaStream_t stream) {
  int m=x.size(0),k=x.size(1),kp=q.size(1)*2,rows=q.size(0);
  int padded=swizzle?(rows+127)/128*128:rows;
  dim3 grid(residue&&!masks?padded/2:padded);
  
  
  
  int threads=(residue&&!masks&&tile8&&m>=64&&(k==4096||k==8192))?k/16:128;
  auto p=reinterpret_cast<T const*>(x.data_ptr());
  #define LAUNCH(R) fused_quant<T,R><<<grid,threads,0,stream>>>(p,x.stride(0),x.stride(1),masks,q.data_ptr<uint8_t>(),sf.data_ptr<uint8_t>(),m,k,kp,rows,padded,swizzle,tile8,pdl_mode)
  if(masks) {switch(per8) {case 1:LAUNCH(1);break;case 2:LAUNCH(2);break;case 4:LAUNCH(4);break;}}
  else if(residue) {LAUNCH(8);} else {LAUNCH(0);}
  #undef LAUNCH
}
} // namespace

void mxfp4_residue_quant(torch::Tensor x,torch::Tensor q,torch::Tensor sf,
         c10::optional<torch::Tensor> masks,int64_t per8,
         bool residue,bool swizzle,bool tile8,int64_t pdl_mode) {
  TORCH_CHECK(pdl_mode>=0 && pdl_mode<=2,"pdl_mode must be 0/off, 1/tail or 2/early");
  TORCH_CHECK(x.is_cuda()&&x.dim()==2,"expected CUDA [M,K]");
  c10::cuda::CUDAGuard guard(x.device());
  auto prop=at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(prop->major==10&&prop->minor==0,"SM100 required");
  TORCH_CHECK(q.is_cuda()&&sf.is_cuda()&&q.device()==x.device()&&sf.device()==x.device(),"output device mismatch");
  TORCH_CHECK(q.scalar_type()==at::kByte&&sf.scalar_type()==at::kByte&&q.is_contiguous()&&sf.is_contiguous(),"contiguous uint8 outputs required");
  int m=x.size(0),k=x.size(1);
  TORCH_CHECK(k>0&&k%32==0,"invalid MXFP4 extent");
  TORCH_CHECK(!masks||(residue&&(per8==1||per8==2||per8==4)),"per8 must be 1/2/4");
  TORCH_CHECK(!tile8 || (residue && !masks && swizzle),
              "tile8 requires full residue and 128x4 scales");
  if(masks) TORCH_CHECK(masks->device()==x.device()&&masks->scalar_type()==at::kByte&&masks->is_contiguous()&&masks->dim()==1&&masks->numel()==k/8,"prepared uint8 masks [K/8] required");
  int rows=q.size(0),selected=masks?k/8*per8:0;
  TORCH_CHECK(q.dim()==2 && (tile8 ? (rows>=2*m && rows%16==0)
                                  : rows==m*(residue&&!masks?2:1)),
              "invalid output rows");
  int kp=q.size(1)*2;
  TORCH_CHECK(kp%32==0&&(masks?kp>=k+selected:kp==k),"invalid output K");
  int64_t ns=swizzle?int64_t((rows+127)/128)*((kp+127)/128)*512:int64_t(rows)*(kp/32);
  TORCH_CHECK(sf.numel()==ns,"invalid SF extent");
  if(!m)return;
  auto stream=at::cuda::getCurrentCUDAStream();
  auto mp=masks?masks->data_ptr<uint8_t>():nullptr;
  #define DTYPE(T) dispatch<T>(x,q,sf,mp,per8,residue,swizzle,tile8,pdl_mode,stream)
  if(x.scalar_type()==at::kHalf){DTYPE(half);}else if(x.scalar_type()==at::kBFloat16){DTYPE(__nv_bfloat16);}
  else{TORCH_CHECK(x.scalar_type()==at::kFloat,"fp16/bf16/fp32 required");DTYPE(float);}
  #undef DTYPE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
