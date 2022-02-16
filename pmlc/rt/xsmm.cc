// Copyright 2020 Intel Corporation

#include "mlir/ExecutionEngine/RunnerUtils.h"

#include "libxsmm.h" // NOLINT [build/include_subdir]

#include "pmlc/rt/symbol_registry.h"
#include "pmlc/util/logging.h"

static constexpr unsigned int NO_BCAST = 0;
static constexpr unsigned int ROW_BCAST = 1;
static constexpr unsigned int COL_BCAST = 2;
static constexpr unsigned int SCALAR_BCAST = 3;

using FunctionPtr = void (*)(const void *, const void *, void *);

extern "C" void plaidml_rt_xsmm_gemm_invoke_f32(int64_t funcAddr, float *a,
                                                float *b, float *c) {
  libxsmm_xmmfunction sgemm;
  sgemm.xmm = reinterpret_cast<FunctionPtr>(funcAddr);
  sgemm.smm(b, a, c);
}

extern "C" int64_t plaidml_rt_xsmm_gemm_dispatch_f32(int32_t lda, int32_t ldb,
                                                     int32_t ldc, int32_t m,
                                                     int32_t n, int32_t k) {
  libxsmm_blasint lda_int = lda;
  libxsmm_blasint ldb_int = ldb;
  libxsmm_blasint ldc_int = ldc;
  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;

  auto sgemm =
      libxsmm_smmdispatch(n_int, m_int, k_int, &ldb_int, &lda_int, &ldc_int,
                          /*alpha=*/nullptr, /*beta=*/nullptr,
                          /*flags=*/nullptr, /*prefetch=*/nullptr);

  return reinterpret_cast<int64_t>(sgemm);
}

extern "C" void plaidml_rt_xsmm_brgemm_invoke_f32(int64_t funcAddr, float *a,
                                                  float *b, float *c,
                                                  int64_t numBatches) {
  libxsmm_xmmfunction sgemm;
  sgemm.xmm = reinterpret_cast<FunctionPtr>(funcAddr);
  unsigned long long numBatchesVar = numBatches; // NOLINT
  sgemm.smrs(b, a, c, &numBatchesVar);
}

extern "C" int64_t plaidml_rt_xsmm_brgemm_dispatch_f32(int32_t lda, int32_t ldb,
                                                       int32_t ldc, int32_t m,
                                                       int32_t n, int32_t k) {
  libxsmm_blasint lda_int = lda;
  libxsmm_blasint ldb_int = ldb;
  libxsmm_blasint ldc_int = ldc;
  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;
  libxsmm_blasint stride_a = k * sizeof(float);
  libxsmm_blasint stride_b = ldb * k * sizeof(float);

  auto sgemm = libxsmm_smmdispatch_reducebatch_strd(
      n_int, m_int, k_int, stride_b, stride_a, &ldb_int, &lda_int, &ldc_int,
      /*alpha=*/nullptr, /*beta=*/nullptr,
      /*flags=*/nullptr, /*prefetch=*/nullptr);

  return reinterpret_cast<int64_t>(sgemm);
}

extern "C" void plaidml_rt_xsmm_brgemm_offs_invoke_f32(
    int64_t funcAddr, float *a, float *b, float *c, int64_t numBatches,
    uint64_t *a_offsets, uint64_t *b_offsets) {
  libxsmm_xmmfunction sgemm;
  sgemm.xmm = reinterpret_cast<FunctionPtr>(funcAddr);
  unsigned long long numBatchesVar = numBatches;                      // NOLINT
  auto *l_a_offs = reinterpret_cast<unsigned long long *>(a_offsets); // NOLINT
  auto *l_b_offs = reinterpret_cast<unsigned long long *>(b_offsets); // NOLINT

  sgemm.smro(b, a, c, &numBatchesVar, l_b_offs, l_a_offs);
}

extern "C" int64_t
plaidml_rt_xsmm_brgemm_offs_dispatch_f32(int32_t lda, int32_t ldb, int32_t ldc,
                                         int32_t m, int32_t n, int32_t k) {
  libxsmm_blasint lda_int = lda;
  libxsmm_blasint ldb_int = ldb;
  libxsmm_blasint ldc_int = ldc;
  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;

  auto sgemm = libxsmm_smmdispatch_reducebatch_offs(
      n_int, m_int, k_int, &ldb_int, &lda_int, &ldc_int,
      /*alpha=*/nullptr, /*beta=*/nullptr,
      /*flags=*/nullptr, /*prefetch=*/nullptr);

  return reinterpret_cast<int64_t>(sgemm);
}

extern "C" int64_t plaidml_rt_xsmm_unary_dispatch(
    int32_t m, int32_t n, int32_t ldi, int32_t ldo, int32_t in_type,
    int32_t compute_type, int32_t out_type, int32_t type, int32_t bcast_type) {
  libxsmm_blasint ldi_int = ldi;
  libxsmm_blasint ldo_int = ldo;
  unsigned int use_bcast = (unsigned int)bcast_type;
  libxsmm_meltw_unary_flags unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;

  if (use_bcast == ROW_BCAST) {
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW;
  } else if (use_bcast == COL_BCAST) {
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL;
  } else if (use_bcast == SCALAR_BCAST) {
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BCAST_SCALAR;
  }
  libxsmm_meltwfunction_unary kernel = libxsmm_dispatch_meltw_unary(
      static_cast<libxsmm_blasint>(n), static_cast<libxsmm_blasint>(m),
      &ldi_int, &ldo_int, // leading dimensions
      static_cast<libxsmm_datatype>(in_type),
      static_cast<libxsmm_datatype>(compute_type),
      static_cast<libxsmm_datatype>(out_type),
      unary_flags, // TODO: add flags to op definition
      static_cast<libxsmm_meltw_unary_type>(type));
  return reinterpret_cast<int64_t>(kernel);
}

extern "C" void plaidml_rt_xsmm_unary_invoke(int64_t addr, void *input,
                                             void *output) {
  libxsmm_meltwfunction_unary kernel =
      reinterpret_cast<libxsmm_meltwfunction_unary>(addr);
  libxsmm_meltw_unary_param param;
  param.in.primary = input;
  param.out.primary = output;
  kernel(&param);
}

extern "C" int64_t plaidml_rt_xsmm_binary_dispatch(
    int32_t m, int32_t n, int32_t ldi1, int32_t ldi2, int32_t ldo,
    int32_t in_type1, int32_t in_type2, int32_t compute_type, int32_t out_type,
    int32_t type, int32_t bcast_type1, int32_t bcast_type2) {
  libxsmm_blasint ldi1_int = ldi1;
  libxsmm_blasint ldi2_int = ldi2;
  libxsmm_blasint ldo_int = ldo;
  libxsmm_meltw_binary_flags binary_flags = LIBXSMM_MELTW_FLAG_BINARY_NONE;

  unsigned int use_bcast1 = (unsigned int)bcast_type1;
  unsigned int use_bcast2 = (unsigned int)bcast_type2;

  if (use_bcast1 == ROW_BCAST) {
    binary_flags = static_cast<libxsmm_meltw_binary_flags>(
        static_cast<int>(binary_flags) |
        static_cast<int>(LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_0));
  } else if (use_bcast1 == COL_BCAST) {
    binary_flags = static_cast<libxsmm_meltw_binary_flags>(
        static_cast<int>(binary_flags) |
        static_cast<int>(LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0));
  } else if (use_bcast1 == SCALAR_BCAST) {
    binary_flags = static_cast<libxsmm_meltw_binary_flags>(
        static_cast<int>(binary_flags) |
        static_cast<int>(LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0));
  }

  if (use_bcast2 == ROW_BCAST) {
    binary_flags = static_cast<libxsmm_meltw_binary_flags>(
        static_cast<int>(binary_flags) |
        static_cast<int>(LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1));
  } else if (use_bcast2 == COL_BCAST) {
    binary_flags = static_cast<libxsmm_meltw_binary_flags>(
        static_cast<int>(binary_flags) |
        static_cast<int>(LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_1));
  } else if (use_bcast2 == SCALAR_BCAST) {
    binary_flags = static_cast<libxsmm_meltw_binary_flags>(
        static_cast<int>(binary_flags) |
        static_cast<int>(LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1));
  }

  libxsmm_meltwfunction_binary kernel = libxsmm_dispatch_meltw_binary(
      static_cast<libxsmm_blasint>(n), static_cast<libxsmm_blasint>(m),
      &ldi1_int, &ldi2_int, &ldo_int, // leading dimensions
      static_cast<libxsmm_datatype>(in_type1),
      static_cast<libxsmm_datatype>(in_type2),
      static_cast<libxsmm_datatype>(out_type), binary_flags,
      static_cast<libxsmm_meltw_binary_type>(type));
  return reinterpret_cast<int64_t>(kernel);
}

extern "C" void plaidml_rt_xsmm_binary_invoke(int64_t addr, void *input1,
                                              void *input2, void *output) {
  libxsmm_meltwfunction_binary kernel =
      reinterpret_cast<libxsmm_meltwfunction_binary>(addr);
  libxsmm_meltw_binary_param param;
  param.in0.primary = input1;
  param.in1.primary = input2;
  param.out.primary = output;
  kernel(&param);
}

namespace pmlc::rt {

void registerXsmm() {
  libxsmm_init();

  REGISTER_SYMBOL(plaidml_rt_xsmm_gemm_invoke_f32);
  REGISTER_SYMBOL(plaidml_rt_xsmm_gemm_dispatch_f32);
  REGISTER_SYMBOL(plaidml_rt_xsmm_brgemm_invoke_f32);
  REGISTER_SYMBOL(plaidml_rt_xsmm_brgemm_dispatch_f32);
  REGISTER_SYMBOL(plaidml_rt_xsmm_brgemm_offs_invoke_f32);
  REGISTER_SYMBOL(plaidml_rt_xsmm_brgemm_offs_dispatch_f32);
  REGISTER_SYMBOL(plaidml_rt_xsmm_unary_dispatch);
  REGISTER_SYMBOL(plaidml_rt_xsmm_unary_invoke);
  REGISTER_SYMBOL(plaidml_rt_xsmm_binary_dispatch);
  REGISTER_SYMBOL(plaidml_rt_xsmm_binary_invoke);
}

} // namespace pmlc::rt
