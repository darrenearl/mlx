// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/qmm/qmm.h"

#include <cute/tensor.hpp>


namespace mlx::core {

namespace {

inline bool is_last_2_dims_row_contiguous(const array& x) {
  return x.flags().contiguous && (x.ndim() >= 2) && (x.strides(-1) == 1) &&
      (x.strides(-2) == x.shape(-1));
}

} // namespace

#if defined(MLX_CUDA_SM90A_ENABLED)
// Defined in qmm_impl_sm90_xxx.cu files.
template <typename TileShape, typename ClusterShape>
void qmm_impl_sm90(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    array& out,
    int bits,
    int group_size,
    cu::CommandEncoder& encoder,
    Stream s);
#endif // defined(MLX_CUDA_SM90A_ENABLED)

bool supports_qmm_sm90(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::Device& device) {
  if (device.compute_capability_major() != 9) {
    return false;
  }
  int k = x.shape(-1);
  if (k % 64 != 0) {
    return false;
  }
  if (!biases) {
    return false;
  }
  if (!x.flags().row_contiguous || !is_last_2_dims_row_contiguous(w) ||
      !is_last_2_dims_row_contiguous(scales) ||
      !is_last_2_dims_row_contiguous(*biases)) {
    return false;
  }
  if (!transpose) {
    return false;
  }
  if (bits % 2 != 0) {
    return false;
  }
  if (group_size < k) {
    return false;
  }
  if (mode != QuantizationMode::Affine) {
    return false;
  }
  return true;
}

void qmm_sm90(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    array& out,
    int bits,
    int group_size,
    cu::CommandEncoder& encoder,
    Stream s) {
#if defined(MLX_CUDA_SM90A_ENABLED)
  auto dispatch = [&]<int tile_m, int tile_n, int cluster_m>() {
    using cute::Int;
    using TileShapeMN = cute::Shape<Int<tile_m>, Int<tile_n>>;
    using ClusterShape = cute::Shape<Int<cluster_m>, Int<1>, Int<1>>;
    qmm_impl_sm90<TileShapeMN, ClusterShape>(
        x, w, scales, biases, out, bits, group_size, encoder, s);
  };
  int m = out.ndim() > 1 ? out.shape(-2) : 1;
  if (m <= 16) {
    dispatch.template operator()<128, 16, 1>();
  } else if (m <= 32) {
    dispatch.template operator()<128, 32, 1>();
  } else if (m <= 64) {
    dispatch.template operator()<128, 64, 2>();
  } else if (m <= 128) {
    dispatch.template operator()<128, 128, 2>();
  } else {
    dispatch.template operator()<128, 256, 2>();
  }
#else
  throw std::runtime_error(
      "[quantized_matmul] Hopper-only kernel is not available.");
#endif // defined(MLX_CUDA_SM90A_ENABLED)
}

// Defined in qmm_impl_sm80_xxx.cu files.
template <int TileM>
void qmm_impl_sm80(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& lhs_indices,
    const std::optional<array>& rhs_indices,
    array& out,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder);

bool supports_qmm_sm80(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::Device& device) {
  if (device.compute_capability_major() < 8) {
    return false;
  }
  int n = out.shape(-1);
  int k = x.shape(-1);
  if ((n % 128 != 0) || (k % std::max(64, group_size) != 0)) {
    return false;
  }
  if (!x.flags().row_contiguous || !is_last_2_dims_row_contiguous(w) ||
      !is_last_2_dims_row_contiguous(scales)) {
    return false;
  }
  if (biases && !is_last_2_dims_row_contiguous(*biases)) {
    return false;
  }
  if (x.dtype() != float16 && x.dtype() != bfloat16) {
    return false;
  }
  if (!transpose) {
    return false;
  }
  if (bits != 4 && bits != 8) {
    return false;
  }
  return true;
}

void qmm_sm80(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& lhs_indices,
    const std::optional<array>& rhs_indices,
    array& out,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder) {
  auto dispatch = [&]<int TileM>() {
    qmm_impl_sm80<TileM>(
        x,
        w,
        scales,
        biases,
        lhs_indices,
        rhs_indices,
        out,
        bits,
        group_size,
        mode,
        encoder);
  };
  int m = out.ndim() > 1 ? out.shape(-2) : 1;
  if (m <= 16) {
    dispatch.template operator()<16>();
  } else if (m <= 32) {
    dispatch.template operator()<32>();
  } else {
    dispatch.template operator()<64>();
  }
}

// Defined in qmm_impl_naive_xxx.cu files.
template <int TileM, bool KMajor>
void qmm_impl_naive(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& lhs_indices,
    const std::optional<array>& rhs_indices,
    array& out,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder);

bool supports_qmm_naive(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::Device& device) {
  int k = x.shape(-1);
  if (transpose && (k % std::max(64, group_size) != 0)) {
    return false;
  }
  if (!x.flags().row_contiguous || !is_last_2_dims_row_contiguous(w) ||
      !is_last_2_dims_row_contiguous(scales)) {
    return false;
  }
  if (biases && !is_last_2_dims_row_contiguous(*biases)) {
    return false;
  }
  return true;
}
// ---------------------------------------------------------------------------
// GQA / multi-dim batch support helpers
// ---------------------------------------------------------------------------
// The underlying qmm kernel encodes batches as a single flat L dim plus one
// bool broadcast_b. For ndim > 3 with mixed broadcast / independent batch
// dims (GQA 5D) this flattening is wrong. Fix: pre-compute a per-L index
// table and feed it via rhs_indices, which the kernel already supports.
//
// WARNING: by the time we get here, MLX may have already materialized w as
// a broadcast *view* — logical shape becomes (1,4,4,...) but strides[2]==0.
// So we must detect broadcast via strides, not shape.
// ---------------------------------------------------------------------------

static std::vector<int> physical_batch_shape(const array& a, int bnd) {
  std::vector<int> bshape(bnd);
  const auto& strides = a.strides();
  for (int i = 0; i < bnd; ++i) {
    if (a.shape(i) > 1 && strides[i] == 0) {
      bshape[i] = 1;
    } else {
      bshape[i] = a.shape(i);
    }
  }
  return bshape;
}

static std::vector<uint32_t> build_batch_index_table(
    const std::vector<int>& full_bshape,
    const std::vector<int>& self_bshape,
    int l) {
  const int bnd = static_cast<int>(full_bshape.size());
  std::vector<uint32_t> table(l);
  std::vector<int> mi(bnd);
  for (int lc = 0; lc < l; ++lc) {
    int tmp = lc;
    for (int i = bnd - 1; i >= 0; --i) {
      mi[i] = tmp % full_bshape[i];
      tmp /= full_bshape[i];
    }
    int flat = 0;
    for (int i = 0; i < bnd; ++i) {
      int coord = (self_bshape[i] == 1) ? 0 : mi[i];
      flat = flat * self_bshape[i] + coord;
    }
    table[lc] = static_cast<uint32_t>(flat);
  }
  return table;
}

static bool is_identity_table(const std::vector<uint32_t>& t) {
  for (size_t i = 0; i < t.size(); ++i) {
    if (t[i] != static_cast<uint32_t>(i)) return false;
  }
  return true;
}

// Upload a host uint32 vector as a device array.
// If the constructor signature differs in your MLX version, adjust here.
static array upload_index_array(const std::vector<uint32_t>& buf) {
  array a(buf.data(), Shape{static_cast<int>(buf.size())}, uint32);
  a.eval();
  return a;
}

void qmm_naive(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const std::optional<array>& lhs_indices,
    const std::optional<array>& rhs_indices,
    array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder) {


  // ---------------- auto-build indices for ndim > 3 ------------------------
  std::optional<array> lhs_idx_local;
  std::optional<array> rhs_idx_local;
  const std::optional<array>* lhs_idx_eff = &lhs_indices;
  const std::optional<array>* rhs_idx_eff = &rhs_indices;

  const int nd = x.ndim();
  const bool user_supplied = lhs_indices.has_value() || rhs_indices.has_value();

  if (nd > 3 && nd == w.ndim() && !user_supplied) {
    const int bnd = nd - 2;
    auto x_bshape = physical_batch_shape(x, bnd);
    auto w_bshape = physical_batch_shape(w, bnd);

    std::vector<int> full_bshape(bnd);
    bool ok = true;
    int l_full = 1;
    for (int i = 0; i < bnd; ++i) {
      int xb = x_bshape[i], wb = w_bshape[i];
      if (xb == wb) full_bshape[i] = xb;
      else if (xb == 1) full_bshape[i] = wb;
      else if (wb == 1) full_bshape[i] = xb;
      else { ok = false; break; }
      l_full *= full_bshape[i];
    }

    if (ok) {
      int m_g = out.ndim() > 1 ? out.shape(-2) : 1;
      int n_g = out.shape(-1);
      int l_out = out.size() / (m_g * n_g);
      if (l_out == l_full) {
        auto lhs_tbl = build_batch_index_table(full_bshape, x_bshape, l_full);
        auto rhs_tbl = build_batch_index_table(full_bshape, w_bshape, l_full);

        if (!is_identity_table(lhs_tbl)) {
          lhs_idx_local = upload_index_array(lhs_tbl);
          lhs_idx_eff = &lhs_idx_local;
        }
        if (!is_identity_table(rhs_tbl)) {
          rhs_idx_local = upload_index_array(rhs_tbl);
          rhs_idx_eff = &rhs_idx_local;
        }
      }
    }
  }
  // -------------------------------------------------------------------------

  auto dispatch = [&]<int TileM, bool KMajor>() {
    qmm_impl_naive<TileM, KMajor>(
        x, w, scales, biases,
        *lhs_idx_eff, *rhs_idx_eff,
        out, bits, group_size, mode, encoder);
  };

  dispatch_bool(transpose, [&](auto k_major) {
    int m = out.ndim() > 1 ? out.shape(-2) : 1;
    if (m <= 16) {
      dispatch.template operator()<16, k_major.value>();
    } else if (m <= 32) {
      dispatch.template operator()<32, k_major.value>();
    } else {
      dispatch.template operator()<64, k_major.value>();
    }
  });
}

bool supports_fp_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::Device& device) {
  // The fp_qmv kernel uses less registers and is faster for sm120. For sm80/90
  // the qmv kernel is faster. We didn't test sm89/100.
  if (device.compute_capability_major() <= 9) {
    return false;
  }
  bool non_batched = w.ndim() == 2;
  int k = x.shape(-1);
  int n = out.shape(-1);
  int vec_batch = non_batched ? x.size() / k : x.shape(-2);
  if (vec_batch > 8) {
    return false;
  }
  if (!transpose) {
    return false;
  }
  if (mode == QuantizationMode::Affine) {
    return false;
  }
  return true;
}

bool supports_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::Device& device) {
  int k = x.shape(-1);
  if (k % 8 != 0) {
    return false;
  }
  if (!x.flags().row_contiguous || !is_last_2_dims_row_contiguous(w) ||
      !is_last_2_dims_row_contiguous(scales)) {
    return false;
  }
  if (biases && !is_last_2_dims_row_contiguous(*biases)) {
    return false;
  }
  if (!transpose) {
    return false;
  }
  return true;
}

} // namespace mlx::core
