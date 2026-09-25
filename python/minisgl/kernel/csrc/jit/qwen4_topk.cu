// SPDX-License-Identifier: Apache-2.0
// Adapted from sgl-project/sglang 5c154c214df569403cb42c5dbb874e4f865aa379,
// kernels/jit/csrc/elementwise/fast_topk.cuh. Radix collection order retained.
#include <cuda_fp16.h>
#include <minisgl/utils.cuh>
#include <tvm/ffi/container/tensor.h>
namespace {

namespace fast_topk_detail {

constexpr uint32_t kThreadsPerBlock = 1024;
// Each radix pass needs at most ~kTopK candidates in the threshold bin, so
// 4K entries per round (2 rounds = 8K entries = 32KB) is sufficient.
constexpr size_t kSmemBytes = 8 * 1024 * sizeof(uint32_t);  // 32KB

struct FastTopKParams {
  const float* __restrict__ input;         // [B, input_stride]
  const int32_t* __restrict__ row_starts;  // [B]
  int32_t* __restrict__ indices;           // [B, kTopK]
  const int32_t* __restrict__ lengths;     // [B]
  int64_t input_stride;
};

__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
  const __half h = __float2half_rn(x);
  const uint16_t bits = __half_as_ushort(h);
  const uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ auto convert_to_uint32(float x) -> uint32_t {
  const uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// When length <= kTopK, write the indices directly.
template <int kTopK>
__device__ __forceinline__ void naive_topk(int32_t* __restrict__ indice, int32_t length) {
  const auto tid = threadIdx.x;
  for (int i = tid; i < kTopK; i += kThreadsPerBlock) {
    indice[i] = (i < length) ? i : -1;
  }
}

// Radix-select top-k. Assumes length > kTopK (checked by the caller).
template <int kTopK>
__device__ __forceinline__ void radix_select_topk(const float* __restrict__ input, int* __restrict__ index, int row_start, int length) {
  int topk = kTopK;
  constexpr auto BLOCK_SIZE = kThreadsPerBlock;
  constexpr auto RADIX = 256;
  constexpr auto SMEM_INPUT_SIZE = kSmemBytes / (2 * sizeof(int));

  alignas(128) __shared__ int s_histogram_buf[2][RADIX + 128];
  alignas(128) __shared__ int s_counter;
  alignas(128) __shared__ int s_threshold_bin_id;
  alignas(128) __shared__ int s_num_input[2];

  auto& s_histogram = s_histogram_buf[0];
  // allocate for two rounds
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];

  const int tx = threadIdx.x;

  // stage 1: 8bit coarse histogram
  if (tx < RADIX + 1) s_histogram[tx] = 0;
  __syncthreads();

  for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
    const auto bin = convert_to_uint8(input[idx + row_start]);
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();

  const auto run_cumsum = [&] {
#pragma unroll 8
    for (int i = 0; i < 8; ++i) {
      static_assert(1 << 8 == RADIX);
      if (tx < RADIX) {
        const auto j = 1 << i;
        const auto k = i & 1;
        auto value = s_histogram_buf[k][tx];
        if (tx < RADIX - j) {
          value += s_histogram_buf[k][tx + j];
        }
        s_histogram_buf[k ^ 1][tx] = value;
      }
      __syncthreads();
    }
  };

  run_cumsum();
  if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
    s_threshold_bin_id = tx;
    s_num_input[0] = 0;
    s_counter = 0;
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  topk -= s_histogram[threshold_bin + 1];

  if (topk == 0) {
    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto bin = static_cast<int>(convert_to_uint8(input[idx + row_start]));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      }
    }
    __syncthreads();
    return;
  } else {
    __syncthreads();
    if (tx < RADIX + 1) {
      s_histogram[tx] = 0;
    }
    __syncthreads();

    for (int idx = tx; idx < length; idx += BLOCK_SIZE) {
      const auto raw_input = input[idx + row_start];
      const auto bin = static_cast<int>(convert_to_uint8(raw_input));
      if (bin > threshold_bin) {
        const auto pos = ::atomicAdd(&s_counter, 1);
        index[pos] = idx;
      } else if (bin == threshold_bin) {
        const auto pos = ::atomicAdd(&s_num_input[0], 1);
        // fuse the histogram computation here
        if (pos < int(SMEM_INPUT_SIZE)) {
          s_input_idx[0][pos] = idx;
          const auto bin = convert_to_uint32(raw_input);
          const auto sub_bin = (bin >> 24) & 0xFF;
          ::atomicAdd(&s_histogram[sub_bin], 1);
        }
      }
    }
    __syncthreads();
  }

  // stage 2: refine with 8bit radix passes
#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    __shared__ int s_last_remain;
    const auto r_idx = round % 2;

    // clip here to prevent overflow
    const auto _raw_num_input = s_num_input[r_idx];
    const auto num_input = (_raw_num_input < int(SMEM_INPUT_SIZE)) ? _raw_num_input : int(SMEM_INPUT_SIZE);

    run_cumsum();
    if (tx < RADIX && s_histogram[tx] > topk && s_histogram[tx + 1] <= topk) {
      s_threshold_bin_id = tx;
      s_num_input[r_idx ^ 1] = 0;
      s_last_remain = topk - s_histogram[tx + 1];
    }
    __syncthreads();

    const auto threshold_bin = s_threshold_bin_id;
    topk -= s_histogram[threshold_bin + 1];

    if (topk == 0) {
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(input[idx + row_start]) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        }
      }
      __syncthreads();
      break;
    } else {
      __syncthreads();
      if (tx < RADIX + 1) {
        s_histogram[tx] = 0;
      }
      __syncthreads();
      for (int i = tx; i < num_input; i += BLOCK_SIZE) {
        const auto idx = s_input_idx[r_idx][i];
        const auto raw_input = input[idx + row_start];
        const auto offset = 24 - round * 8;
        const auto bin = (convert_to_uint32(raw_input) >> offset) & 0xFF;
        if (bin > threshold_bin) {
          const auto pos = ::atomicAdd(&s_counter, 1);
          index[pos] = idx;
        } else if (bin == threshold_bin) {
          if (round == 3) {
            const auto pos = ::atomicAdd(&s_last_remain, -1);
            if (pos > 0) {
              index[kTopK - pos] = idx;
            }
          } else {
            const auto pos = ::atomicAdd(&s_num_input[r_idx ^ 1], 1);
            if (pos < int(SMEM_INPUT_SIZE)) {
              // fuse the histogram computation here
              s_input_idx[r_idx ^ 1][pos] = idx;
              const auto bin = convert_to_uint32(raw_input);
              const auto sub_bin = (bin >> (offset - 8)) & 0xFF;
              ::atomicAdd(&s_histogram[sub_bin], 1);
            }
          }
        }
      }
      __syncthreads();
    }
  }
}

template <int kTopK, bool kUsePDL>
__global__ __launch_bounds__(fast_topk_detail::kThreadsPerBlock) void fast_topk_kernel(
    const fast_topk_detail::FastTopKParams __grid_constant__ params) {
  using namespace fast_topk_detail;


  const auto bid = static_cast<uint64_t>(blockIdx.x);
  const auto row_start = params.row_starts[bid];
  const auto length = params.lengths[bid];
  const auto indice = params.indices + bid * kTopK;
  const auto score = params.input + bid * params.input_stride;
  if (length <= kTopK) {
    naive_topk<kTopK>(indice, length);
  } else {
    radix_select_topk<kTopK>(score, indice, row_start, length);
  }


}

}  // namespace fast_topk_detail

struct Qwen4FastTopK {
  static void run(tvm::ffi::TensorView score, tvm::ffi::TensorView starts,
                  tvm::ffi::TensorView indices, tvm::ffi::TensorView lengths) {
    const auto params = fast_topk_detail::FastTopKParams{
      .input = static_cast<const float*>(score.data_ptr()),
      .row_starts = static_cast<const int32_t*>(starts.data_ptr()),
      .indices = static_cast<int32_t*>(indices.data_ptr()),
      .lengths = static_cast<const int32_t*>(lengths.data_ptr()),
      .input_stride = score.stride(0),
    };
    host::LaunchKernel(score.size(0), fast_topk_detail::kThreadsPerBlock,
                       score.device(), fast_topk_detail::kSmemBytes)(
        fast_topk_detail::fast_topk_kernel<512, false>, params);
  }
};
}
