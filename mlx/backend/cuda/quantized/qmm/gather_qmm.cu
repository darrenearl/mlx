// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/qmm/qmm.h"
#include "mlx/dtype_utils.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cute/numeric/numeric_types.hpp>
#include <cutlass/numeric_conversion.h>

// uint3b_t and uint5b_t are not provided by cutlass, define them here.
namespace cutlass {

using uint3b_t = integer_subbyte<3, false>;
using uint5b_t = integer_subbyte<5, false>;

template <typename T, int N, FloatRoundStyle Round>
struct NumericArrayConverter<T, uint3b_t, N, Round> {
  static_assert(N % 8 == 0);
  using result_type = Array<T, N>;
  using source_type = Array<uint3b_t, N>;
  CUTLASS_HOST_DEVICE
  static result_type convert(const source_type& source) {
    result_type result;
    auto* s_base = reinterpret_cast<const uint8_t*>(&source);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 8; ++i) {
      auto* s = s_base + i * 3;
      result[i * 8] = T(s[0] & 0x07);
      result[i * 8 + 1] = T((s[0] & 0x38) >> 3);
      result[i * 8 + 2] = T((s[0] & 0xc0) >> 6) + T((s[1] & 0x01) << 2);
      result[i * 8 + 3] = T((s[1] & 0x0e) >> 1);
      result[i * 8 + 4] = T((s[1] & 0x70) >> 4);
      result[i * 8 + 5] = T((s[1] & 0x80) >> 7) + T((s[2] & 0x03) << 1);
      result[i * 8 + 6] = T((s[2] & 0x1c) >> 2);
      result[i * 8 + 7] = T((s[2] & 0xe0) >> 5);
    }
    return result;
  }
  CUTLASS_HOST_DEVICE
  result_type operator()(const source_type& s) const { return convert(s); }
};

template <typename T, int N, FloatRoundStyle Round>
struct NumericArrayConverter<T, uint5b_t, N, Round> {
  static_assert(N % 8 == 0);
  using result_type = Array<T, N>;
  using source_type = Array<uint5b_t, N>;
  CUTLASS_HOST_DEVICE
  static result_type convert(const source_type& source) {
    result_type result;
    auto* s_base = reinterpret_cast<const uint8_t*>(&source);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 8; ++i) {
      auto* s = s_base + i * 5;
      result[i * 8] = T(s[0] & 0x1f);
      result[i * 8 + 1] = T((s[0] & 0xe0) >> 5) + T((s[1] & 0x03) << 3);
      result[i * 8 + 2] = T((s[1] & 0x7c) >> 2);
      result[i * 8 + 3] = T((s[1] & 0x80) >> 7) + T((s[2] & 0x0f) << 1);
      result[i * 8 + 4] = T((s[2] & 0xf0) >> 4) + T((s[3] & 0x01) << 4);
      result[i * 8 + 5] = T((s[3] & 0x3e) >> 1);
      result[i * 8 + 6] = T((s[3] & 0xc0) >> 6) + T((s[4] & 0x07) << 2);
      result[i * 8 + 7] = T((s[4] & 0xf8) >> 3);
    }
    return result;
  }
  CUTLASS_HOST_DEVICE
  result_type operator()(const source_type& s) const { return convert(s); }
};

template <typename T, int N, FloatRoundStyle Round>
struct NumericArrayConverter<T, uint6b_t, N, Round> {
  static_assert(N % 4 == 0);
  using result_type = Array<T, N>;
  using source_type = Array<uint6b_t, N>;
  CUTLASS_HOST_DEVICE
  static result_type convert(const source_type& source) {
    result_type result;
    auto* s_base = reinterpret_cast<const uint8_t*>(&source);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 4; ++i) {
      auto* s = s_base + i * 3;
      result[i * 4] = T(s[0] & 0x3f);
      result[i * 4 + 1] = T((s[0] >> 6) & 0x03) + T((s[1] & 0x0f) << 2);
      result[i * 4 + 2] = T((s[1] >> 4) & 0x0f) + T((s[2] & 0x03) << 4);
      result[i * 4 + 3] = T((s[2] >> 2) & 0x3f);
    }
    return result;
  }
  CUTLASS_HOST_DEVICE
  result_type operator()(const source_type& s) const { return convert(s); }
};

} // namespace cutlass

namespace mlx::core {

template <typename F>
inline void dispatch_element_types(Dtype dtype, const char* tag, F&& f) {
  if (dtype == float32) {
    f.template operator()<float>();
  } else if (dtype == float16) {
    f.template operator()<cutlass::half_t>();
  } else if (dtype == bfloat16) {
    f.template operator()<cutlass::bfloat16_t>();
  } else {
    throw std::invalid_argument(
        fmt::format("{} Unsupported dtype: {}.", tag, dtype_to_string(dtype)));
  }
}

template <typename F>
inline void dispatch_groups(int group_size, const char* tag, F&& f) {
  if (group_size == 32) {
    f.template operator()<32>();
  } else if (group_size == 64) {
    f.template operator()<64>();
  } else if (group_size == 128) {
    f.template operator()<128>();
  } else {
    throw std::invalid_argument(
        fmt::format("{} Group size {} is not supported.", tag, group_size));
  }
}

namespace cu {

namespace cg = cooperative_groups;

// Fused vectorized dequantize and multiply-add.
template <int N, bool has_bias, typename T, typename Q, typename S>
__device__ __forceinline__ void
dequant_fma(const T* x, const Q* w, S scale, T bias, T* out) {
  auto x_vec = *(reinterpret_cast<const cutlass::Array<T, N>*>(x));
  auto w_vec = *(reinterpret_cast<const cutlass::Array<Q, N>*>(w));
  auto* out_vec = reinterpret_cast<cutlass::Array<T, N>*>(out);
  cutlass::NumericArrayConverter<T, Q, N> converter_tq;
  cutlass::Array<T, N> w_dq = converter_tq(w_vec);
  if constexpr (has_bias) {
    if constexpr (cuda::std::is_same_v<T, float>) {
#pragma unroll
      for (int i = 0; i < N; ++i) {
        w_dq[i] = w_dq[i] * T(scale) + bias;
      }
    } else {
      w_dq = w_dq * T(scale) + bias;
    }
  } else {
    w_dq = w_dq * T(scale);
  }
  *out_vec = cutlass::fma(x_vec, w_dq, *out_vec);
}

// Specialization for float32 accumulations on narrow types.
template <
    int N,
    bool has_bias,
    typename T,
    typename Q,
    typename S,
    typename = cuda::std::enable_if_t<!cuda::std::is_same_v<T, float>>>
__device__ __forceinline__ void
dequant_fma(const T* x, const Q* w, S scale, T bias, float* out) {
  auto x_vec = *(reinterpret_cast<const cutlass::Array<T, N>*>(x));
  auto w_vec = *(reinterpret_cast<const cutlass::Array<Q, N>*>(w));
  auto* out_vec = reinterpret_cast<cutlass::Array<float, N>*>(out);
  cutlass::NumericArrayConverter<T, Q, N> converter_tq;
  cutlass::Array<T, N> w_dq = converter_tq(w_vec);
  if constexpr (has_bias) {
    w_dq = w_dq * T(scale) + bias;
  } else {
    w_dq = w_dq * T(scale);
  }
  static_assert(!cuda::std::is_same_v<T, float>);
  cutlass::NumericArrayConverter<float, T, N> converter_ft;
  cutlass::Array<float, N> x_f = converter_ft(x_vec);
  cutlass::Array<float, N> w_f = converter_ft(w_dq);
  *out_vec = cutlass::fma(x_f, w_f, *out_vec);
}

// ---------------------------------------------------------------------------
// gather_qmv_kernel
//
// For MoE-style gather quantized matmul.
//
// x layout:     [x_batch..., M, K]       -- flattened to [x_els, M, K]
// w layout:     [num_experts, N, K_packed]
// scales:       [num_experts, N, groups_per_row]
// biases:       [num_experts, N, groups_per_row]
// lhs_indices:  [batch_size] (flattened)  -- index into x's batch dims
// rhs_indices:  [batch_size] (flattened)  -- index into w's expert dim
// out layout:   [batch_size, M, N]        (flattened output batch)
//
// Grid: {ceil_div(N, rows_per_block) * batch_size, M, 1}
// Block: {WARP_SIZE, rows_per_block}
//
// NOTE: batch_size is folded into grid.x (instead of grid.z) to avoid
// the 65535 limit on grid.z/grid.y when batch_size >= 65536.
// ---------------------------------------------------------------------------
template <
    int rows_per_block,
    int elems_per_thread,
    int group_size,
    bool has_bias,
    bool has_residue_k,
    typename T,
    typename Q,
    typename S>
__global__ void gather_qmv_kernel(
    const T* x,
    const Q* w,
    const S* scales,
    const T* biases,
    const uint32_t* lhs_indices,
    const uint32_t* rhs_indices,
    T* out,
    int n,
    int k,
    int batch_size) {
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  // Decode batch_idx and row from blockIdx.x
  // grid.x = n_blocks_per_batch * batch_size
  int n_blocks_per_batch = cuda::ceil_div(n, rows_per_block);
  int batch_idx = block.group_index().x / n_blocks_per_batch;
  int row = (block.group_index().x % n_blocks_per_batch) * rows_per_block
            + warp.meta_group_rank();

  if (row >= n) {
    return;
  }

  int m = gridDim.y;

  // Gather indices.
  uint32_t lhs_idx = lhs_indices[batch_idx];
  uint32_t rhs_idx = rhs_indices[batch_idx];

  // x: [x_els, m, k] -> advance to the right batch element and m row.
  const T* x_ptr = x + static_cast<int64_t>(lhs_idx) * m * k +
      block.group_index().y * k;

  // out: [batch_size, m, n] -> advance to right batch and m row.
  T* out_ptr = out + static_cast<int64_t>(batch_idx) * m * n +
      block.group_index().y * n;

  // Sub-byte pointer arithmetic helper.
  constexpr int bits = cute::sizeof_bits_v<Q>;
  auto w_step = [&](int idx) { return idx * cuda::std::min(8, bits) / 8; };

  int groups_per_row = k / group_size;

  // w: [num_experts, n, k_packed] -> advance to expert and row.
  const Q* w_row = w + static_cast<int64_t>(rhs_idx) * n * w_step(k) +
      static_cast<int64_t>(row) * w_step(k);
  const S* scales_row =
      scales + static_cast<int64_t>(rhs_idx) * n * groups_per_row +
      static_cast<int64_t>(row) * groups_per_row;
  const T* biases_row = nullptr;
  if constexpr (has_bias) {
    biases_row =
        biases + static_cast<int64_t>(rhs_idx) * n * groups_per_row +
        static_cast<int64_t>(row) * groups_per_row;
  }

  // Accumulations.
  cuda::std::conditional_t<(bits >= 8), float, T> sums[elems_per_thread] = {};

  auto dequant_fma_tile = [&](int idx) {
    S scale = scales_row[idx / group_size];
    T bias{0};
    if constexpr (has_bias) {
      bias = biases_row[idx / group_size];
    }
    dequant_fma<elems_per_thread, has_bias>(
        x_ptr + idx, w_row + w_step(idx), scale, bias, sums);
  };

  constexpr int elems_per_warp = WARP_SIZE * elems_per_thread;
  for (int r = 0; r < k / elems_per_warp; ++r) {
    int idx = warp.thread_rank() * elems_per_thread + r * elems_per_warp;
    dequant_fma_tile(idx);
  }

  if constexpr (has_residue_k) {
    int rest = k % elems_per_warp;
    int idx = warp.thread_rank() * elems_per_thread + k - rest;
    if (idx < k) {
      dequant_fma_tile(idx);
    }
  }

  float sum{0};
#pragma unroll
  for (int i = 0; i < elems_per_thread; ++i) {
    sum += sums[i];
  }
  sum = cg::reduce(warp, sum, cg::plus<float>{});

  if (warp.thread_rank() == 0) {
    out_ptr[row] = static_cast<T>(sum);
  }
}

// ---------------------------------------------------------------------------
// gather_qmv launch wrapper
// ---------------------------------------------------------------------------
template <
    int group_size,
    bool has_bias,
    typename T,
    typename Q,
    typename S,
    typename F>
void gather_qmv(
    const T* x,
    const Q* w,
    const S* scales,
    const T* biases,
    const uint32_t* lhs_indices,
    const uint32_t* rhs_indices,
    T* out,
    int m,
    int n,
    int k,
    int batch_size,
    F&& launch_kernel) {
  constexpr int rows_per_block = 8;
  constexpr int elems_per_thread =
      (cute::sizeof_bits_v<T> <= 16 && cute::sizeof_bits_v<Q> <= 4) ? 16 : 8;

  // Fold batch_size into grid.x to avoid the 65535 limit on grid.z.
  // grid.x = ceil_div(n, rows_per_block) * batch_size
  // Kernel decodes: batch_idx = blockIdx.x / n_blocks_per_batch
  //                 block_n   = blockIdx.x % n_blocks_per_batch
  dim3 num_blocks{
      uint32_t(cuda::ceil_div(n, rows_per_block)) * uint32_t(batch_size),
      uint32_t(m),
      1};
  dim3 block_dims{WARP_SIZE, rows_per_block};
  void* args[] = {
      &x,
      &w,
      &scales,
      &biases,
      &lhs_indices,
      &rhs_indices,
      &out,
      &n,
      &k,
      &batch_size};

  dispatch_bool(k % (WARP_SIZE * elems_per_thread), [&](auto has_residue_k) {
    auto* kernel = &gather_qmv_kernel<
        rows_per_block,
        elems_per_thread,
        group_size,
        has_bias,
        has_residue_k.value,
        T,
        Q,
        S>;
    launch_kernel(
        reinterpret_cast<void*>(kernel), num_blocks, block_dims, args);
  });
}

// ---------------------------------------------------------------------------
// gather_qmm_tiled_kernel - Tiled GEMM for large M
// ---------------------------------------------------------------------------
template<
    int TILE_M, int TILE_N, int TILE_K, int VEC_SIZE,
    int group_size, bool has_bias,
    typename T, typename Q, typename S>
__global__ void gather_qmm_tiled_kernel(
    const T* __restrict__ x,
    const Q* __restrict__ w,
    const S* __restrict__ scales,
    const T* __restrict__ biases,
    const uint32_t* __restrict__ lhs_indices,
    const uint32_t* __restrict__ rhs_indices,
    T* __restrict__ out,
    int M, int N, int K, int batch_size) {
  // Decode batch_idx and n_start from blockIdx.x
  // grid.x = n_blocks_per_batch * batch_size
  int n_blocks_per_batch = (N + TILE_N - 1) / TILE_N;
  int batch_idx = blockIdx.x / n_blocks_per_batch;
  int n_start = (blockIdx.x % n_blocks_per_batch) * TILE_N;
  int m_start = blockIdx.y * TILE_M;

  uint32_t lhs_idx = lhs_indices[batch_idx];
  uint32_t rhs_idx = rhs_indices[batch_idx];

  const T* x_base = x + static_cast<int64_t>(lhs_idx) * M * K;
  T* out_base = out + static_cast<int64_t>(batch_idx) * M * N;

  constexpr int bits = cute::sizeof_bits_v<Q>;
  auto w_step = [](int idx) -> int { return idx * cuda::std::min(8, bits) / 8; };

  int groups_per_row = K / group_size;
  const Q* w_base = w + static_cast<int64_t>(rhs_idx) * N * w_step(K);
  const S* scales_base = scales + static_cast<int64_t>(rhs_idx) * N * groups_per_row;
  const T* biases_base = nullptr;
  if constexpr (has_bias) {
    biases_base = biases + static_cast<int64_t>(rhs_idx) * N * groups_per_row;
  }

  extern __shared__ char shared_mem[];
  T* smem_x = reinterpret_cast<T*>(shared_mem);
  T* smem_w = smem_x + TILE_M * TILE_K;

  constexpr int THREADS_N = TILE_N / 4;
  constexpr int NUM_THREADS = TILE_M * THREADS_N;
  int tid = threadIdx.x;
  int thread_m = tid / THREADS_N;
  int thread_n = (tid % THREADS_N) * 4;

  float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  int k_tiles = (K + TILE_K - 1) / TILE_K;
  for (int kt = 0; kt < k_tiles; ++kt) {
    int k_start = kt * TILE_K;
    int k_len = min(TILE_K, K - k_start);

    // Load X tile
    {
      int total_x = TILE_M * TILE_K;
      for (int idx = tid; idx < total_x; idx += NUM_THREADS) {
        int row = idx / TILE_K;
        int col = idx % TILE_K;
        int gm = m_start + row;
        int gk = k_start + col;
        smem_x[row * TILE_K + col] =
            (gm < M && col < k_len) ? x_base[gm * K + gk] : T(0);
      }
    }

    // Load & dequantize W tile
    {
      constexpr int qbits = cute::sizeof_bits_v<Q>;
      int total_w = TILE_N * TILE_K;
      for (int idx = tid; idx < total_w; idx += NUM_THREADS) {
        int row_n = idx / TILE_K;
        int col_k = idx % TILE_K;
        int gn = n_start + row_n;
        int gk = k_start + col_k;
        if (gn < N && col_k < k_len) {
          int gi = gk / group_size;
          S scale = scales_base[gn * groups_per_row + gi];

          // Bit-level extraction for any sub-byte or byte quantized type.
          int64_t bit_idx = static_cast<int64_t>(gn) * K + gk;
          const uint8_t* raw_base = reinterpret_cast<const uint8_t*>(w_base);
          int64_t byte_pos = bit_idx * qbits / 8;
          int bit_offset = (bit_idx * qbits) % 8;
          uint8_t mask = static_cast<uint8_t>((1u << qbits) - 1u);

          uint16_t twobytes;
          if constexpr (qbits <= 8) {
            // Read up to 2 bytes to handle cross-byte boundaries.
            twobytes = raw_base[byte_pos];
            if (bit_offset + qbits > 8) {
              twobytes |= static_cast<uint16_t>(raw_base[byte_pos + 1]) << 8;
            }
          }
          T val = T(int((twobytes >> bit_offset) & mask)) * T(scale);
          if constexpr (has_bias) {
            val = val + biases_base[gn * groups_per_row + gi];
          }
          smem_w[row_n * TILE_K + col_k] = val;
        } else {
          smem_w[row_n * TILE_K + col_k] = T(0);
        }
      }
    }

    __syncthreads();

    if (thread_m < TILE_M && (m_start + thread_m) < M) {
      const T* my_x = smem_x + thread_m * TILE_K;
#pragma unroll 8
      for (int kk = 0; kk < k_len; ++kk) {
        float xv = float(my_x[kk]);
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          accum[ni] += xv * float(smem_w[(thread_n + ni) * TILE_K + kk]);
        }
      }
    }
    __syncthreads();
  }

  if (thread_m < TILE_M && (m_start + thread_m) < M) {
    int gm = m_start + thread_m;
#pragma unroll
    for (int ni = 0; ni < 4; ++ni) {
      int gn = n_start + thread_n + ni;
      if (gn < N) {
        out_base[gm * N + gn] = T(accum[ni]);
      }
    }
  }
}

template <int group_size, bool has_bias,
          typename T, typename Q, typename S, typename F>
void gather_qmm_tiled(
    const T* x, const Q* w, const S* scales, const T* biases,
    const uint32_t* lhs_indices, const uint32_t* rhs_indices,
    T* out, int m, int n, int k, int batch_size,
    F&& launch_kernel) {
  constexpr int TILE_M = 32;
  constexpr int TILE_N = 64;
  constexpr int TILE_K = 64;
  constexpr int VEC_SIZE = 8;
  constexpr int NUM_THREADS = TILE_M * (TILE_N / 4);
  size_t smem_bytes = (TILE_M * TILE_K + TILE_N * TILE_K) * sizeof(T);

  // Fold batch_size into grid.x to avoid the 65535 limit on grid.z.
  // grid.x = n_blocks_per_batch * batch_size
  // Kernel decodes: batch_idx = blockIdx.x / n_blocks_per_batch
  //                 block_n   = blockIdx.x % n_blocks_per_batch
  uint32_t n_blocks_per_batch = uint32_t(cuda::ceil_div(n, TILE_N));
  dim3 num_blocks{
      n_blocks_per_batch * uint32_t(batch_size),
      uint32_t(cuda::ceil_div(m, TILE_M)),
      1};
  dim3 block_dims{NUM_THREADS};
  void* args[] = {
      &x, &w, &scales, &biases,
      &lhs_indices, &rhs_indices, &out,
      &m, &n, &k, &batch_size};

  auto* kernel = &gather_qmm_tiled_kernel<
      TILE_M, TILE_N, TILE_K, VEC_SIZE,
      group_size, has_bias, T, Q, S>;

  launch_kernel(reinterpret_cast<void*>(kernel),
                num_blocks, block_dims, smem_bytes, args);
}

} // namespace cu

// ---------------------------------------------------------------------------
// gather_qmm entry point
// ---------------------------------------------------------------------------
void gather_qmm(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    bool transpose_,
    int group_size_,
    int bits_,
    int M,
    int N,
    int K,
    cu::CommandEncoder& enc,
    const Stream& s) {
  const char* tag = "[gather_qmm]";

  if (!transpose_) {
    throw std::runtime_error(fmt::format(
        "{} Non-transposed gather_qmm is not yet supported on CUDA.", tag));
  }

  // batch_size = total number of elements in indices (flattened).
  // e.g. indices shape [1, 1, 8] -> batch_size = 8
  int batch_size = lhs_indices.size();

  // For M > 4, use tiled GEMM kernel for better performance.
  bool use_tiled = (M > 4);

  dispatch_element_types(out.dtype(), tag, [&]<typename T>() {
    dispatch_groups(group_size_, tag, [&]<int group_size>() {
      auto dispatch_bits = [&]<typename Q>() {
        using S = T;
        constexpr bool has_bias = true;

        enc.set_input_array(x);
        enc.set_input_array(w);
        enc.set_input_array(scales);
        enc.set_input_array(biases);
        enc.set_input_array(lhs_indices);
        enc.set_input_array(rhs_indices);
        enc.set_output_array(out);

        if (use_tiled) {
          // Tiled GEMM path: W is loaded into shared memory once and
          // reused across TILE_M rows of X, greatly reducing global
          // memory bandwidth for large M.
          cu::gather_qmm_tiled<group_size, has_bias>(
              gpu_ptr<T>(x),
              gpu_ptr<Q>(w),
              gpu_ptr<S>(scales),
              gpu_ptr<T>(biases),
              gpu_ptr<uint32_t>(lhs_indices),
              gpu_ptr<uint32_t>(rhs_indices),
              gpu_ptr<T>(out),
              M,
              N,
              K,
              batch_size,
              [&](auto* kernel,
                  dim3 num_blocks,
                  dim3 block_dims,
                  uint32_t smem_bytes,
                  void** args) {
                enc.add_kernel_node_raw(
                    kernel, num_blocks, block_dims, {}, smem_bytes, args);
              });
        } else {
          // Original matvec path: optimal for M <= 4 where each warp
          // computes one dot product along K.
          cu::gather_qmv<group_size, has_bias>(
              gpu_ptr<T>(x),
              gpu_ptr<Q>(w),
              gpu_ptr<S>(scales),
              gpu_ptr<T>(biases),
              gpu_ptr<uint32_t>(lhs_indices),
              gpu_ptr<uint32_t>(rhs_indices),
              gpu_ptr<T>(out),
              M,
              N,
              K,
              batch_size,
              [&](auto* kernel,
                  dim3 num_blocks,
                  dim3 block_dims,
                  void** args) {
                enc.add_kernel_node_raw(
                    kernel, num_blocks, block_dims, {}, 0, args);
              });
        }
      };

      if (bits_ == 2) {
        dispatch_bits.template operator()<cutlass::uint2b_t>();
      } else if (bits_ == 3) {
        dispatch_bits.template operator()<cutlass::uint3b_t>();
      } else if (bits_ == 4) {
        dispatch_bits.template operator()<cutlass::uint4b_t>();
      } else if (bits_ == 5) {
        dispatch_bits.template operator()<cutlass::uint5b_t>();
      } else if (bits_ == 6) {
        dispatch_bits.template operator()<cutlass::uint6b_t>();
      } else if (bits_ == 8) {
        dispatch_bits.template operator()<uint8_t>();
      } else {
        throw std::invalid_argument(fmt::format(
            "{} {}-bit quantization is not supported.", tag, bits_));
      }
    });
  });
}


} // namespace mlx::core
