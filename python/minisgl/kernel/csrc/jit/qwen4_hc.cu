// SPDX-License-Identifier: Apache-2.0
// Arithmetic/reduction order follows SGLang's grouped_gemma_rmsnorm.cuh and
// hc_combine.cuh (sgl-project/sglang, Apache-2.0), implemented with mini's JIT.
#include <cuda_bf16.h>
#include <minisgl/utils.cuh>
#include <tvm/ffi/container/tensor.h>

namespace {
using Bf16 = __nv_bfloat16;
union alignas(16) Pack8 { uint4 bits; Bf16 values[8]; };

__device__ __forceinline__ Pack8 load8(const Bf16* p) {
  Pack8 v;
  v.bits = *reinterpret_cast<const uint4*>(p);
  return v;
}

__device__ __forceinline__ void store8(Bf16* p, const Pack8& v) {
  *reinterpret_cast<uint4*>(p) = v.bits;
}

__device__ __forceinline__ float warp_sum(float value) {
  #pragma unroll
  for (int mask = 16; mask >= 1; mask >>= 1)
    value += __shfl_xor_sync(0xffffffff, value, mask);
  return value;
}

template<int D, int HC>
__global__ __launch_bounds__(D / 16) void hc_norm(
    const Bf16* x, const Bf16* w, Bf16* out, float eps) {
  constexpr int NW = D / 16 / 32;
  __shared__ float sums[32];
  int row = blockIdx.x, tid = threadIdx.x;
  float values[16], weights[16];
  #pragma unroll
  for (int j = 0; j < 2; ++j) {
    Pack8 xv = load8(x + row * D + tid * 16 + j * 8);
    Pack8 wv = load8(w + (row % HC) * D + tid * 16 + j * 8);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      values[j * 8 + i] = __bfloat162float(xv.values[i]);
      weights[j * 8 + i] = __bfloat162float(wv.values[i]);
    }
  }
  float sum = 0;
  #pragma unroll
  for (int i = 0; i < 8; ++i)
    sum += values[2*i] * values[2*i] + values[2*i+1] * values[2*i+1];
  sum = warp_sum(sum);
  int warp = tid / 32;
  if (tid % 32 == 0) sums[warp] = sum;
  __syncthreads();
  if (warp == 0) {
    sum = warp_sum(tid < NW ? sums[tid] : 0.f);
    sums[tid] = rsqrtf(sum / D + eps);
  }
  __syncthreads();
  float inv = sums[warp];
  #pragma unroll
  for (int j = 0; j < 2; ++j) {
    Pack8 ov;
    #pragma unroll
    for (int i = 0; i < 8; ++i)
      ov.values[i] = __float2bfloat16_rn(values[j*8+i] * inv * (1.f + weights[j*8+i]));
    store8(out + row * D + tid * 16 + j * 8, ov);
  }
}

template<int H, int HC>
__global__ __launch_bounds__(256) void hc_combine(
    const Bf16* residual, const Bf16* x, const Bf16* norm,
    const Bf16* weight, Bf16* out) {
  constexpr int K = H * HC, N = K / (256 * 8);
  __shared__ float sums[HC][8], gates[HC];
  int row = blockIdx.x, tid = threadIdx.x;
  Pack8 norms[N];
  #pragma unroll
  for (int j = 0; j < N; ++j)
    norms[j] = load8(norm + row * K + (tid + j * 256) * 8);
  #pragma unroll
  for (int c = 0; c < HC; ++c) {
    float sum = 0.f;
    #pragma unroll
    for (int j = 0; j < N; ++j) {
      int start = (tid + j * 256) * 8;
      Pack8 wv = load8(weight + c * K + start);
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        float nx = __bfloat162float(norms[j].values[2*i]);
        float ny = __bfloat162float(norms[j].values[2*i+1]);
        float wx = __bfloat162float(wv.values[2*i]);
        float wy = __bfloat162float(wv.values[2*i+1]);
        sum += nx * wx + ny * wy;
      }
    }
    sum = warp_sum(sum);
    if (tid % 32 == 0) sums[c][tid / 32] = sum;
  }
  __syncthreads();
  if (tid < HC) {
    float sum = 0.f;
    #pragma unroll
    for (int warp = 0; warp < 8; ++warp) sum += sums[tid][warp];
    gates[tid] = 2.f / (1.f + expf(-sum / HC));
  }
  __syncthreads();
  #pragma unroll
  for (int j = 0; j < N; ++j) {
    int start = (tid + j * 256) * 8;
    float gate = gates[start / H];
    Pack8 rv = load8(residual + row * K + start);
    Pack8 xv = load8(x + row * H + start % H), ov;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      float r = __bfloat162float(rv.values[i]);
      float value = __bfloat162float(xv.values[i]);
      ov.values[i] = __float2bfloat16_rn(r + gate * value);
    }
    store8(out + row * K + start, ov);
  }
}

template<int H, int HC>
__global__ __launch_bounds__(32) void hc_gate_split(
    const Bf16* norm, const Bf16* weight, float* partials) {
  constexpr int K = H * HC, N = K / (256 * 8);
  int row = blockIdx.x, split = blockIdx.y / HC, c = blockIdx.y % HC;
  int tid = split * 32 + threadIdx.x;
  float sum = 0.f;
  #pragma unroll
  for (int j = 0; j < N; ++j) {
    int start = (tid + j * 256) * 8;
    Pack8 nv = load8(norm + row*K + start), wv = load8(weight + c*K + start);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      float nx = __bfloat162float(nv.values[2*i]), ny = __bfloat162float(nv.values[2*i+1]);
      float wx = __bfloat162float(wv.values[2*i]), wy = __bfloat162float(wv.values[2*i+1]);
      sum += nx * wx + ny * wy;
    }
  }
  sum = warp_sum(sum);
  if (threadIdx.x == 0) partials[(row*8 + split)*HC + c] = sum;
}

template<int H, int HC>
__global__ void hc_apply_split(const Bf16* residual, const Bf16* x,
                              const float* partials, Bf16* out) {
  constexpr int K = H * HC, VECTORS = K / 64;
  int row = blockIdx.x, at = (blockIdx.y * VECTORS + threadIdx.x) * 8;
  int branch = at / H;
  float total = 0.f;
  #pragma unroll
  for (int s = 0; s < 8; ++s) total += partials[(row*8+s)*HC+branch];
  float gate = 2.f / (1.f + expf(-total / HC));
  Pack8 rv = load8(residual + row*K + at), xv = load8(x + row*H + at%H), ov;
  #pragma unroll
  for (int i = 0; i < 8; ++i)
    ov.values[i] = __float2bfloat16_rn(__bfloat162float(rv.values[i]) + gate * __bfloat162float(xv.values[i]));
  store8(out + row*K + at, ov);
}

template<int H, int HC> struct Qwen4HC {
  static void combine_split(tvm::ffi::TensorView r, tvm::ffi::TensorView x,
                            tvm::ffi::TensorView n, tvm::ffi::TensorView w,
                            tvm::ffi::TensorView y, tvm::ffi::TensorView p) {
    static_assert(H * HC / 64 <= 1024);
    host::LaunchKernel(dim3(x.size(0), 8 * HC), 32, x.device())(
        hc_gate_split<H, HC>, static_cast<const Bf16*>(n.data_ptr()),
        static_cast<const Bf16*>(w.data_ptr()), static_cast<float*>(p.data_ptr()));
    host::LaunchKernel(dim3(x.size(0), 8), H * HC / 64, x.device())(
        hc_apply_split<H, HC>, static_cast<const Bf16*>(r.data_ptr()),
        static_cast<const Bf16*>(x.data_ptr()), static_cast<const float*>(p.data_ptr()),
        static_cast<Bf16*>(y.data_ptr()));
  }
  static void norm(tvm::ffi::TensorView x, tvm::ffi::TensorView w,
                   tvm::ffi::TensorView y, float eps) {
    host::LaunchKernel(x.numel() / H, H / 16, x.device())(
        hc_norm<H, HC>, static_cast<const Bf16*>(x.data_ptr()),
        static_cast<const Bf16*>(w.data_ptr()), static_cast<Bf16*>(y.data_ptr()), eps);
  }
  static void combine(tvm::ffi::TensorView r, tvm::ffi::TensorView x,
                      tvm::ffi::TensorView n, tvm::ffi::TensorView w,
                      tvm::ffi::TensorView y) {
    static_assert(H * HC % 2048 == 0);
    host::LaunchKernel(x.size(0), 256, x.device())(
        hc_combine<H, HC>, static_cast<const Bf16*>(r.data_ptr()),
        static_cast<const Bf16*>(x.data_ptr()), static_cast<const Bf16*>(n.data_ptr()),
        static_cast<const Bf16*>(w.data_ptr()), static_cast<Bf16*>(y.data_ptr()));
  }
};

// SGLang qsa_indexer.cuh: indexer RoPE intentionally rounds every operation
// to BF16, unlike the attention Q/K fused RoPE which retains FP32 products.
template<int D>
__global__ void index_norm_rope(const Bf16* x, const Bf16* w, const float* cache,
                              const int64_t* positions, Bf16* out,
                              int heads, int rd, float eps) {
  __shared__ Bf16 normed[D];
  int row = blockIdx.x, lane = threadIdx.x;
  float sum = 0.f;
  if (lane < D / 8) {
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      float v = __bfloat162float(x[row * D + lane * 8 + j]);
      sum += v * v;
    }
  }
  float inv = rsqrtf(warp_sum(sum) / D + eps);
  #pragma unroll
  for (int j = 0; j < D / 32; ++j) {
    int d = lane * (D / 32) + j;
    float v = __bfloat162float(x[row * D + d]);
    float weight = __bfloat162float(w[d]);
    normed[d] = __float2bfloat16_rn(v * inv * (1.f + weight));
  }
  __syncwarp();
  int64_t pos = positions[row / heads];
  #pragma unroll
  for (int j = 0; j < D / 32; ++j) {
    int d = lane * (D / 32) + j;
    Bf16 result = normed[d];
    if (d < rd) {
      int pair = d % (rd / 2), peer = d < rd / 2 ? d + rd / 2 : d - rd / 2;
      float co = __bfloat162float(__float2bfloat16_rn(cache[pos * rd + pair]));
      float si = __bfloat162float(__float2bfloat16_rn(cache[pos * rd + rd / 2 + pair]));
      float a = __bfloat162float(__float2bfloat16_rn(__bfloat162float(normed[d]) * co));
      float b = __bfloat162float(__float2bfloat16_rn(__bfloat162float(normed[peer]) * si));
      result = __float2bfloat16_rn(d < rd / 2 ? a - b : a + b);
    }
    out[row * D + d] = result;
  }
}

template<int D> struct Qwen4Index {
  static void run(tvm::ffi::TensorView x, tvm::ffi::TensorView w,
                  tvm::ffi::TensorView cache, tvm::ffi::TensorView pos,
                  tvm::ffi::TensorView out, float eps) {
    host::LaunchKernel(x.numel() / D, 32, x.device())(
        index_norm_rope<D>, static_cast<const Bf16*>(x.data_ptr()),
        static_cast<const Bf16*>(w.data_ptr()), static_cast<const float*>(cache.data_ptr()),
        static_cast<const int64_t*>(pos.data_ptr()), static_cast<Bf16*>(out.data_ptr()),
        static_cast<int>(x.size(1)), static_cast<int>(cache.size(1)), eps);
  }
};
}
