// Copyright 2020 Intel Corporation

#include "mlir/ExecutionEngine/RunnerUtils.h"

#include "libxsmm.h" // NOLINT [build/include_subdir]

#include "pmlc/rt/symbol_registry.h"
#include "pmlc/util/logging.h"

static constexpr unsigned int NO_BCAST = 0;
static constexpr unsigned int ROW_BCAST = 1;
static constexpr unsigned int COL_BCAST = 2;
static constexpr unsigned int SCALAR_BCAST = 3;
static constexpr unsigned int NO_REDUCE = 0;
static constexpr unsigned int REEDUCE_ROWS = 1;
static constexpr unsigned int REEDUCE_COLS = 2;

using FunctionPtr = void (*)(const void *, const void *, void *);

extern "C" void plaidml_rt_xsmm_gemm_invoke_f32(int64_t funcAddr, float *a,
                                                float *b, float *c) {
  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_param gemm_param;
  gemm_param.a.primary = reinterpret_cast<void *>(b);
  gemm_param.b.primary = reinterpret_cast<void *>(a);
  gemm_param.c.primary = reinterpret_cast<void *>(c);
  sgemm.gemm = reinterpret_cast<libxsmm_gemmfunction>(funcAddr);
  sgemm.gemm(&gemm_param);
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

  libxsmm_gemm_shape l_shape;
  libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  libxsmm_bitfield l_prefetch_flags = 0;

  l_shape.m = n_int;
  l_shape.n = m_int;
  l_shape.k = k_int;
  l_shape.lda = ldb_int;
  l_shape.ldb = lda_int;
  l_shape.ldc = ldc_int;
  l_shape.a_in_type = LIBXSMM_DATATYPE_F32;
  l_shape.b_in_type = LIBXSMM_DATATYPE_F32;
  l_shape.out_type = LIBXSMM_DATATYPE_F32;
  l_shape.comp_type = LIBXSMM_DATATYPE_F32;

  auto sgemm = libxsmm_dispatch_gemm_v2(l_shape, l_flags, l_prefetch_flags);

  return reinterpret_cast<int64_t>(sgemm);
}

extern "C" void plaidml_rt_xsmm_gemm_relu_invoke_f32(int64_t funcAddr, float *a,
                                                     float *b, float *c,
                                                     float *d) {
  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_ext_param gemm_param;
  gemm_param.a.primary = reinterpret_cast<void *>(b);
  gemm_param.b.primary = reinterpret_cast<void *>(a);
  gemm_param.c.primary = reinterpret_cast<void *>(c);
  gemm_param.d.primary = reinterpret_cast<void *>(d);
  sgemm.gemm_ext = reinterpret_cast<libxsmm_gemmfunction_ext>(funcAddr);
  sgemm.gemm_ext(&gemm_param);
}

extern "C" int64_t
plaidml_rt_xsmm_gemm_relu_dispatch_f32(int32_t lda, int32_t ldb, int32_t ldc,
                                       int32_t m, int32_t n, int32_t k) {
  libxsmm_blasint lda_int = lda;
  libxsmm_blasint ldb_int = ldb;
  libxsmm_blasint ldc_int = ldc;
  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;

  libxsmm_gemm_shape l_shape;
  libxsmm_bitfield l_flags =
      LIBXSMM_GEMM_FLAGS('N', 'N') | LIBXSMM_GEMM_FLAG_BETA_0;
  libxsmm_bitfield l_prefetch_flags = 0;

  l_shape.m = n_int;
  l_shape.n = m_int;
  l_shape.k = k_int;
  l_shape.lda = ldb_int;
  l_shape.ldb = lda_int;
  l_shape.ldc = ldc_int;
  l_shape.a_in_type = LIBXSMM_DATATYPE_F32;
  l_shape.b_in_type = LIBXSMM_DATATYPE_F32;
  l_shape.out_type = LIBXSMM_DATATYPE_F32;
  l_shape.comp_type = LIBXSMM_DATATYPE_F32;

  libxsmm_gemm_batch_reduce_config l_brconfig;
  l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_NONE;

  libxsmm_gemm_ext_unary_argops l_argops;
  memset(&l_argops, 0, sizeof(libxsmm_gemm_ext_unary_argops));
  l_argops.ldcp = ldc;
  l_argops.cp_unary_type = LIBXSMM_MELTW_TYPE_UNARY_RELU;

  libxsmm_gemm_ext_binary_postops l_postops;
  memset(&l_postops, 0, sizeof(libxsmm_gemm_ext_binary_postops));
  l_postops.d_in_type = LIBXSMM_DATATYPE_F32;
  l_postops.d_binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
  l_postops.d_binary_type = LIBXSMM_MELTW_TYPE_BINARY_ADD;
  l_postops.ldd = ldc;

  auto sgemm = libxsmm_dispatch_brgemm_ext_v2(
      l_shape, l_flags, l_prefetch_flags, l_brconfig, l_argops, l_postops);

  return reinterpret_cast<int64_t>(sgemm);
}

extern "C" void plaidml_rt_xsmm_gemm_bias_invoke_f32(int64_t funcAddr, float *a,
                                                     float *b, float *c,
                                                     float *d) {
  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_ext_param gemm_param;
  gemm_param.a.primary = reinterpret_cast<void *>(b);
  gemm_param.b.primary = reinterpret_cast<void *>(a);
  gemm_param.c.primary = reinterpret_cast<void *>(c);
  gemm_param.d.primary = reinterpret_cast<void *>(d);
  sgemm.gemm_ext = reinterpret_cast<libxsmm_gemmfunction_ext>(funcAddr);
  sgemm.gemm_ext(&gemm_param);
}

extern "C" int64_t
plaidml_rt_xsmm_gemm_bias_dispatch_f32(int32_t lda, int32_t ldb, int32_t ldc,
                                       int32_t m, int32_t n, int32_t k) {
  libxsmm_blasint lda_int = lda;
  libxsmm_blasint ldb_int = ldb;
  libxsmm_blasint ldc_int = ldc;
  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;

  libxsmm_gemm_shape l_shape;
  libxsmm_bitfield l_flags =
      LIBXSMM_GEMM_FLAGS('N', 'N') | LIBXSMM_GEMM_FLAG_BETA_0;
  libxsmm_bitfield l_prefetch_flags = 0;

  l_shape.m = n_int;
  l_shape.n = m_int;
  l_shape.k = k_int;
  l_shape.lda = ldb_int;
  l_shape.ldb = lda_int;
  l_shape.ldc = ldc_int;
  l_shape.a_in_type = LIBXSMM_DATATYPE_F32;
  l_shape.b_in_type = LIBXSMM_DATATYPE_F32;
  l_shape.out_type = LIBXSMM_DATATYPE_F32;
  l_shape.comp_type = LIBXSMM_DATATYPE_F32;

  libxsmm_gemm_batch_reduce_config l_brconfig;
  l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_NONE;

  libxsmm_gemm_ext_unary_argops l_argops;
  memset(&l_argops, 0, sizeof(libxsmm_gemm_ext_unary_argops));
  l_argops.ldcp = ldc;

  libxsmm_gemm_ext_binary_postops l_postops;
  memset(&l_postops, 0, sizeof(libxsmm_gemm_ext_binary_postops));
  l_postops.d_in_type = LIBXSMM_DATATYPE_F32;
  l_postops.d_binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
  l_postops.d_binary_type = LIBXSMM_MELTW_TYPE_BINARY_ADD;
  l_postops.ldd = ldc;

  auto sgemm = libxsmm_dispatch_brgemm_ext_v2(
      l_shape, l_flags, l_prefetch_flags, l_brconfig, l_argops, l_postops);

  return reinterpret_cast<int64_t>(sgemm);
}

extern "C" void plaidml_rt_xsmm_gemm_relu_beta1_invoke_f32(int64_t funcAddr,
                                                           float *a, float *b,
                                                           float *c, float *d) {
  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_ext_param gemm_param;
  gemm_param.a.primary = reinterpret_cast<void *>(b);
  gemm_param.b.primary = reinterpret_cast<void *>(a);
  gemm_param.c.primary = reinterpret_cast<void *>(c);
  gemm_param.d.primary = reinterpret_cast<void *>(d);
  sgemm.gemm_ext = reinterpret_cast<libxsmm_gemmfunction_ext>(funcAddr);
  sgemm.gemm_ext(&gemm_param);
}

extern "C" int64_t plaidml_rt_xsmm_gemm_relu_beta1_dispatch_f32(
    int32_t lda, int32_t ldb, int32_t ldc, int32_t m, int32_t n, int32_t k) {
  libxsmm_blasint lda_int = lda;
  libxsmm_blasint ldb_int = ldb;
  libxsmm_blasint ldc_int = ldc;
  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;

  libxsmm_gemm_shape l_shape;
  libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  libxsmm_bitfield l_prefetch_flags = 0;

  l_shape.m = n_int;
  l_shape.n = m_int;
  l_shape.k = k_int;
  l_shape.lda = ldb_int;
  l_shape.ldb = lda_int;
  l_shape.ldc = ldc_int;
  l_shape.a_in_type = LIBXSMM_DATATYPE_F32;
  l_shape.b_in_type = LIBXSMM_DATATYPE_F32;
  l_shape.out_type = LIBXSMM_DATATYPE_F32;
  l_shape.comp_type = LIBXSMM_DATATYPE_F32;

  libxsmm_gemm_batch_reduce_config l_brconfig;
  l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_NONE;

  libxsmm_gemm_ext_unary_argops l_argops;
  memset(&l_argops, 0, sizeof(libxsmm_gemm_ext_unary_argops));
  l_argops.ldcp = ldc;
  l_argops.cp_unary_type = LIBXSMM_MELTW_TYPE_UNARY_RELU;

  libxsmm_gemm_ext_binary_postops l_postops;
  memset(&l_postops, 0, sizeof(libxsmm_gemm_ext_binary_postops));
  l_postops.d_in_type = LIBXSMM_DATATYPE_F32;
  l_postops.d_binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
  l_postops.d_binary_type = LIBXSMM_MELTW_TYPE_BINARY_ADD;
  l_postops.ldd = ldc;

  auto sgemm = libxsmm_dispatch_brgemm_ext_v2(
      l_shape, l_flags, l_prefetch_flags, l_brconfig, l_argops, l_postops);

  return reinterpret_cast<int64_t>(sgemm);
}

extern "C" void plaidml_rt_xsmm_brgemm_invoke_f32(int64_t funcAddr, float *a,
                                                  float *b, float *c,
                                                  int64_t numBatches) {
  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_param gemm_param;
  sgemm.gemm = reinterpret_cast<libxsmm_gemmfunction>(funcAddr);
  unsigned long long numBatchesVar = numBatches; // NOLINT
  gemm_param.a.primary = reinterpret_cast<void *>(b);
  gemm_param.b.primary = reinterpret_cast<void *>(a);
  gemm_param.c.primary = reinterpret_cast<void *>(c);
  gemm_param.op.tertiary = reinterpret_cast<void *>(&numBatchesVar);
  sgemm.gemm(&gemm_param);
}

extern "C" int64_t plaidml_rt_xsmm_brgemm_dispatch_f32(int32_t lda, int32_t ldb,
                                                       int32_t ldc, int32_t m,
                                                       int32_t n, int32_t k,
                                                       int64_t stride_a_hint,
                                                       int64_t stride_b_hint) {
  libxsmm_blasint lda_int = lda;
  libxsmm_blasint ldb_int = ldb;
  libxsmm_blasint ldc_int = ldc;
  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;
  libxsmm_blasint stride_a = stride_a_hint;
  libxsmm_blasint stride_b = stride_b_hint;

  libxsmm_gemm_shape l_shape;
  libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  libxsmm_bitfield l_prefetch_flags = 0;
  libxsmm_gemm_batch_reduce_config l_brconfig;

  l_shape.m = n_int;
  l_shape.n = m_int;
  l_shape.k = k_int;
  l_shape.lda = ldb_int;
  l_shape.ldb = lda_int;
  l_shape.ldc = ldc_int;
  l_shape.a_in_type = LIBXSMM_DATATYPE_F32;
  l_shape.b_in_type = LIBXSMM_DATATYPE_F32;
  l_shape.out_type = LIBXSMM_DATATYPE_F32;
  l_shape.comp_type = LIBXSMM_DATATYPE_F32;
  l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
  l_brconfig.br_stride_a_hint = stride_b;
  l_brconfig.br_stride_b_hint = stride_a;
  l_brconfig.br_unroll_hint = 0;

  auto sgemm = libxsmm_dispatch_brgemm_v2(l_shape, l_flags, l_prefetch_flags,
                                          l_brconfig);

  return reinterpret_cast<int64_t>(sgemm);
}

extern "C" void plaidml_rt_xsmm_brgemm_offs_invoke_f32(
    int64_t funcAddr, float *a, float *b, float *c, int64_t numBatches,
    uint64_t *a_offsets, uint64_t *b_offsets) {
  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_param gemm_param;
  sgemm.gemm = reinterpret_cast<libxsmm_gemmfunction>(funcAddr);
  unsigned long long numBatchesVar = numBatches;                      // NOLINT
  auto *l_a_offs = reinterpret_cast<unsigned long long *>(a_offsets); // NOLINT
  auto *l_b_offs = reinterpret_cast<unsigned long long *>(b_offsets); // NOLINT

  gemm_param.a.secondary = reinterpret_cast<void *>(l_b_offs);
  gemm_param.b.secondary = reinterpret_cast<void *>(l_a_offs);
  gemm_param.a.primary = reinterpret_cast<void *>(b);
  gemm_param.b.primary = reinterpret_cast<void *>(a);
  gemm_param.c.primary = reinterpret_cast<void *>(c);
  gemm_param.op.tertiary = reinterpret_cast<void *>(&numBatchesVar);
  sgemm.gemm(&gemm_param);
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

  libxsmm_gemm_shape l_shape;
  libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  libxsmm_bitfield l_prefetch_flags = 0;
  libxsmm_gemm_batch_reduce_config l_brconfig;

  l_shape.m = n_int;
  l_shape.n = m_int;
  l_shape.k = k_int;
  l_shape.lda = ldb_int;
  l_shape.ldb = lda_int;
  l_shape.ldc = ldc_int;
  l_shape.a_in_type = LIBXSMM_DATATYPE_F32;
  l_shape.b_in_type = LIBXSMM_DATATYPE_F32;
  l_shape.out_type = LIBXSMM_DATATYPE_F32;
  l_shape.comp_type = LIBXSMM_DATATYPE_F32;
  l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_OFFSET;
  l_brconfig.br_stride_a_hint = 0;
  l_brconfig.br_stride_b_hint = 0;
  l_brconfig.br_unroll_hint = 0;

  auto sgemm = libxsmm_dispatch_brgemm_v2(l_shape, l_flags, l_prefetch_flags,
                                          l_brconfig);

  return reinterpret_cast<int64_t>(sgemm);
}

extern "C" int64_t
plaidml_rt_xsmm_unary_dispatch(int32_t m, int32_t n, int32_t ldi, int32_t ldo,
                               int32_t in_type, int32_t compute_type,
                               int32_t out_type, int32_t type,
                               int32_t bcast_type, int32_t reduce_type) {
  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint ldi_int = ldi;
  libxsmm_blasint ldo_int = ldo;
  unsigned int use_bcast = (unsigned int)bcast_type;
  unsigned int use_reduce = (unsigned int)reduce_type;
  libxsmm_meltw_unary_flags unary_flags = LIBXSMM_MELTW_FLAG_UNARY_NONE;

  libxsmm_meltw_unary_shape unary_shape;

  if (use_bcast == ROW_BCAST) {
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BCAST_ROW;
  } else if (use_bcast == COL_BCAST) {
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL;
  } else if (use_bcast == SCALAR_BCAST) {
    unary_flags = LIBXSMM_MELTW_FLAG_UNARY_BCAST_SCALAR;
  }

  if (reduce_type == REEDUCE_ROWS) {
    unary_flags = static_cast<libxsmm_meltw_unary_flags>(
        static_cast<int>(LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS) |
        static_cast<int>(LIBXSMM_MELTW_FLAG_UNARY_REDUCE_INIT_ACC));
    ldo = ldi;
  } else if (reduce_type == REEDUCE_COLS) {
    m_int = n;
    n_int = m;
    unary_flags = static_cast<libxsmm_meltw_unary_flags>(
        static_cast<int>(LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS) |
        static_cast<int>(LIBXSMM_MELTW_FLAG_UNARY_REDUCE_INIT_ACC));
    ldo = ldi;
  }

  unary_shape.m = n_int;
  unary_shape.n = m_int;
  unary_shape.in0_type = static_cast<libxsmm_datatype>(in_type);
  unary_shape.comp_type = static_cast<libxsmm_datatype>(compute_type);
  unary_shape.out_type = static_cast<libxsmm_datatype>(out_type);
  unary_shape.ldi = ldi_int;
  unary_shape.ldo = ldo_int;

  libxsmm_meltwfunction_unary kernel = libxsmm_dispatch_meltw_unary_v2(
      static_cast<libxsmm_meltw_unary_type>(type), unary_shape,
      static_cast<libxsmm_bitfield>(unary_flags));

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

  libxsmm_meltw_binary_shape binary_shape;

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

  binary_shape.m = static_cast<libxsmm_blasint>(n);
  binary_shape.n = static_cast<libxsmm_blasint>(m);
  binary_shape.in0_type = static_cast<libxsmm_datatype>(in_type1);
  binary_shape.in1_type = static_cast<libxsmm_datatype>(in_type2);
  binary_shape.comp_type = static_cast<libxsmm_datatype>(compute_type);
  binary_shape.out_type = static_cast<libxsmm_datatype>(out_type);
  binary_shape.ldi = ldi1_int;
  binary_shape.ldi2 = ldi2_int;
  binary_shape.ldo = ldo_int;

  libxsmm_meltwfunction_binary kernel = libxsmm_dispatch_meltw_binary_v2(
      static_cast<libxsmm_meltw_binary_type>(type), binary_shape,
      static_cast<libxsmm_bitfield>(binary_flags));

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

extern "C" void plaidml_rt_xsmm_brgemm_relu_invoke_f32(int64_t funcAddr,
                                                       float *a, float *b,
                                                       float *c, float *d,
                                                       int64_t numBatches) {
  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_ext_param gemm_param;
  sgemm.gemm_ext = reinterpret_cast<libxsmm_gemmfunction_ext>(funcAddr);
  unsigned long long numBatchesVar = numBatches; // NOLINT
  gemm_param.a.primary = reinterpret_cast<void *>(b);
  gemm_param.b.primary = reinterpret_cast<void *>(a);
  gemm_param.c.primary = reinterpret_cast<void *>(c);
  gemm_param.d.primary = reinterpret_cast<void *>(d);
  gemm_param.op.tertiary = reinterpret_cast<void *>(&numBatchesVar);
  sgemm.gemm_ext(&gemm_param);
}

extern "C" void plaidml_rt_xsmm_brgemm_bias_invoke_f32(int64_t funcAddr,
                                                       float *a, float *b,
                                                       float *c, float *d,
                                                       int64_t numBatches) {
  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_ext_param gemm_param;
  sgemm.gemm_ext = reinterpret_cast<libxsmm_gemmfunction_ext>(funcAddr);
  unsigned long long numBatchesVar = numBatches; // NOLINT
  gemm_param.a.primary = reinterpret_cast<void *>(b);
  gemm_param.b.primary = reinterpret_cast<void *>(a);
  gemm_param.c.primary = reinterpret_cast<void *>(c);
  gemm_param.d.primary = reinterpret_cast<void *>(d);
  gemm_param.op.tertiary = reinterpret_cast<void *>(&numBatchesVar);
  sgemm.gemm_ext(&gemm_param);
}

extern "C" void
plaidml_rt_xsmm_brgemm_relu_beta1_invoke_f32(int64_t funcAddr, float *a,
                                             float *b, float *c, float *d,
                                             int64_t numBatches) {
  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_ext_param gemm_param;
  sgemm.gemm_ext = reinterpret_cast<libxsmm_gemmfunction_ext>(funcAddr);
  unsigned long long numBatchesVar = numBatches; // NOLINT
  gemm_param.a.primary = reinterpret_cast<void *>(b);
  gemm_param.b.primary = reinterpret_cast<void *>(a);
  gemm_param.c.primary = reinterpret_cast<void *>(c);
  gemm_param.d.primary = reinterpret_cast<void *>(d);
  gemm_param.op.tertiary = reinterpret_cast<void *>(&numBatchesVar);
  sgemm.gemm_ext(&gemm_param);
}

extern "C" int64_t plaidml_rt_xsmm_brgemm_relu_dispatch_f32(
    int32_t lda, int32_t ldb, int32_t ldc, int32_t m, int32_t n, int32_t k,
    int64_t stride_a_hint, int64_t stride_b_hint) {
  libxsmm_blasint lda_int = lda;
  libxsmm_blasint ldb_int = ldb;
  libxsmm_blasint ldc_int = ldc;
  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;
  libxsmm_blasint stride_a = stride_a_hint;
  libxsmm_blasint stride_b = stride_b_hint;

  libxsmm_gemm_shape l_shape;
  libxsmm_bitfield l_flags =
      LIBXSMM_GEMM_FLAGS('N', 'N') | LIBXSMM_GEMM_FLAG_BETA_0;
  libxsmm_bitfield l_prefetch_flags = 0;
  libxsmm_gemm_batch_reduce_config l_brconfig;

  l_shape.m = n_int;
  l_shape.n = m_int;
  l_shape.k = k_int;
  l_shape.lda = ldb_int;
  l_shape.ldb = lda_int;
  l_shape.ldc = ldc_int;
  l_shape.a_in_type = LIBXSMM_DATATYPE_F32;
  l_shape.b_in_type = LIBXSMM_DATATYPE_F32;
  l_shape.out_type = LIBXSMM_DATATYPE_F32;
  l_shape.comp_type = LIBXSMM_DATATYPE_F32;
  l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
  l_brconfig.br_stride_a_hint = stride_b;
  l_brconfig.br_stride_b_hint = stride_a;
  l_brconfig.br_unroll_hint = 0;

  libxsmm_gemm_ext_unary_argops l_argops;
  memset(&l_argops, 0, sizeof(libxsmm_gemm_ext_unary_argops));
  l_argops.ldcp = ldc;
  l_argops.cp_unary_type = LIBXSMM_MELTW_TYPE_UNARY_RELU;

  libxsmm_gemm_ext_binary_postops l_postops;
  memset(&l_postops, 0, sizeof(libxsmm_gemm_ext_binary_postops));
  l_postops.d_in_type = LIBXSMM_DATATYPE_F32;
  l_postops.d_binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
  l_postops.d_binary_type = LIBXSMM_MELTW_TYPE_BINARY_ADD;
  l_postops.ldd = ldc;

  auto sgemm = libxsmm_dispatch_brgemm_ext_v2(
      l_shape, l_flags, l_prefetch_flags, l_brconfig, l_argops, l_postops);

  return reinterpret_cast<int64_t>(sgemm);
}

extern "C" int64_t plaidml_rt_xsmm_brgemm_bias_dispatch_f32(
    int32_t lda, int32_t ldb, int32_t ldc, int32_t m, int32_t n, int32_t k,
    int64_t stride_a_hint, int64_t stride_b_hint) {
  libxsmm_blasint lda_int = lda;
  libxsmm_blasint ldb_int = ldb;
  libxsmm_blasint ldc_int = ldc;
  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;
  libxsmm_blasint stride_a = stride_a_hint;
  libxsmm_blasint stride_b = stride_b_hint;

  libxsmm_gemm_shape l_shape;
  libxsmm_bitfield l_flags =
      LIBXSMM_GEMM_FLAGS('N', 'N') | LIBXSMM_GEMM_FLAG_BETA_0;
  libxsmm_bitfield l_prefetch_flags = 0;
  libxsmm_gemm_batch_reduce_config l_brconfig;

  l_shape.m = n_int;
  l_shape.n = m_int;
  l_shape.k = k_int;
  l_shape.lda = ldb_int;
  l_shape.ldb = lda_int;
  l_shape.ldc = ldc_int;
  l_shape.a_in_type = LIBXSMM_DATATYPE_F32;
  l_shape.b_in_type = LIBXSMM_DATATYPE_F32;
  l_shape.out_type = LIBXSMM_DATATYPE_F32;
  l_shape.comp_type = LIBXSMM_DATATYPE_F32;
  l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
  l_brconfig.br_stride_a_hint = stride_b;
  l_brconfig.br_stride_b_hint = stride_a;
  l_brconfig.br_unroll_hint = 0;

  libxsmm_gemm_ext_unary_argops l_argops;
  memset(&l_argops, 0, sizeof(libxsmm_gemm_ext_unary_argops));
  l_argops.ldcp = ldc;

  libxsmm_gemm_ext_binary_postops l_postops;
  memset(&l_postops, 0, sizeof(libxsmm_gemm_ext_binary_postops));
  l_postops.d_in_type = LIBXSMM_DATATYPE_F32;
  l_postops.d_binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
  l_postops.d_binary_type = LIBXSMM_MELTW_TYPE_BINARY_ADD;
  l_postops.ldd = ldc;

  auto sgemm = libxsmm_dispatch_brgemm_ext_v2(
      l_shape, l_flags, l_prefetch_flags, l_brconfig, l_argops, l_postops);

  return reinterpret_cast<int64_t>(sgemm);
}

extern "C" int64_t plaidml_rt_xsmm_brgemm_relu_beta1_dispatch_f32(
    int32_t lda, int32_t ldb, int32_t ldc, int32_t m, int32_t n, int32_t k,
    int64_t stride_a_hint, int64_t stride_b_hint) {
  libxsmm_blasint lda_int = lda;
  libxsmm_blasint ldb_int = ldb;
  libxsmm_blasint ldc_int = ldc;
  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;
  libxsmm_blasint stride_a = stride_a_hint;
  libxsmm_blasint stride_b = stride_b_hint;

  libxsmm_gemm_shape l_shape;
  libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  libxsmm_bitfield l_prefetch_flags = 0;
  libxsmm_gemm_batch_reduce_config l_brconfig;

  l_shape.m = n_int;
  l_shape.n = m_int;
  l_shape.k = k_int;
  l_shape.lda = ldb_int;
  l_shape.ldb = lda_int;
  l_shape.ldc = ldc_int;
  l_shape.a_in_type = LIBXSMM_DATATYPE_F32;
  l_shape.b_in_type = LIBXSMM_DATATYPE_F32;
  l_shape.out_type = LIBXSMM_DATATYPE_F32;
  l_shape.comp_type = LIBXSMM_DATATYPE_F32;
  l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_STRIDE;
  l_brconfig.br_stride_a_hint = stride_b;
  l_brconfig.br_stride_b_hint = stride_a;
  l_brconfig.br_unroll_hint = 0;

  libxsmm_gemm_ext_unary_argops l_argops;
  memset(&l_argops, 0, sizeof(libxsmm_gemm_ext_unary_argops));
  l_argops.ldcp = ldc;
  l_argops.cp_unary_type = LIBXSMM_MELTW_TYPE_UNARY_RELU;

  libxsmm_gemm_ext_binary_postops l_postops;
  memset(&l_postops, 0, sizeof(libxsmm_gemm_ext_binary_postops));
  l_postops.d_in_type = LIBXSMM_DATATYPE_F32;
  l_postops.d_binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
  l_postops.d_binary_type = LIBXSMM_MELTW_TYPE_BINARY_ADD;
  l_postops.ldd = ldc;

  auto sgemm = libxsmm_dispatch_brgemm_ext_v2(
      l_shape, l_flags, l_prefetch_flags, l_brconfig, l_argops, l_postops);

  return reinterpret_cast<int64_t>(sgemm);
}

extern "C" void plaidml_rt_xsmm_brgemm_offs_relu_invoke_f32(
    int64_t funcAddr, float *a, float *b, float *c, float *d,
    int64_t numBatches, uint64_t *a_offsets, uint64_t *b_offsets) {
  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_ext_param gemm_param;
  sgemm.gemm_ext = reinterpret_cast<libxsmm_gemmfunction_ext>(funcAddr);
  unsigned long long numBatchesVar = numBatches;                      // NOLINT
  auto *l_a_offs = reinterpret_cast<unsigned long long *>(a_offsets); // NOLINT
  auto *l_b_offs = reinterpret_cast<unsigned long long *>(b_offsets); // NOLINT

  gemm_param.a.secondary = reinterpret_cast<void *>(l_b_offs);
  gemm_param.b.secondary = reinterpret_cast<void *>(l_a_offs);
  gemm_param.a.primary = reinterpret_cast<void *>(b);
  gemm_param.b.primary = reinterpret_cast<void *>(a);
  gemm_param.c.primary = reinterpret_cast<void *>(c);
  gemm_param.d.primary = reinterpret_cast<void *>(d);
  gemm_param.op.tertiary = reinterpret_cast<void *>(&numBatchesVar);
  sgemm.gemm_ext(&gemm_param);
}

extern "C" void plaidml_rt_xsmm_brgemm_offs_bias_invoke_f32(
    int64_t funcAddr, float *a, float *b, float *c, float *d,
    int64_t numBatches, uint64_t *a_offsets, uint64_t *b_offsets) {
  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_ext_param gemm_param;
  sgemm.gemm_ext = reinterpret_cast<libxsmm_gemmfunction_ext>(funcAddr);
  unsigned long long numBatchesVar = numBatches;                      // NOLINT
  auto *l_a_offs = reinterpret_cast<unsigned long long *>(a_offsets); // NOLINT
  auto *l_b_offs = reinterpret_cast<unsigned long long *>(b_offsets); // NOLINT

  gemm_param.a.secondary = reinterpret_cast<void *>(l_b_offs);
  gemm_param.b.secondary = reinterpret_cast<void *>(l_a_offs);
  gemm_param.a.primary = reinterpret_cast<void *>(b);
  gemm_param.b.primary = reinterpret_cast<void *>(a);
  gemm_param.c.primary = reinterpret_cast<void *>(c);
  gemm_param.d.primary = reinterpret_cast<void *>(d);
  gemm_param.op.tertiary = reinterpret_cast<void *>(&numBatchesVar);
  sgemm.gemm_ext(&gemm_param);
}

extern "C" void plaidml_rt_xsmm_brgemm_offs_relu_beta1_invoke_f32(
    int64_t funcAddr, float *a, float *b, float *c, float *d,
    int64_t numBatches, uint64_t *a_offsets, uint64_t *b_offsets) {
  libxsmm_xmmfunction sgemm;
  libxsmm_gemm_ext_param gemm_param;
  sgemm.gemm_ext = reinterpret_cast<libxsmm_gemmfunction_ext>(funcAddr);
  unsigned long long numBatchesVar = numBatches;                      // NOLINT
  auto *l_a_offs = reinterpret_cast<unsigned long long *>(a_offsets); // NOLINT
  auto *l_b_offs = reinterpret_cast<unsigned long long *>(b_offsets); // NOLINT

  gemm_param.a.secondary = reinterpret_cast<void *>(l_b_offs);
  gemm_param.b.secondary = reinterpret_cast<void *>(l_a_offs);
  gemm_param.a.primary = reinterpret_cast<void *>(b);
  gemm_param.b.primary = reinterpret_cast<void *>(a);
  gemm_param.c.primary = reinterpret_cast<void *>(c);
  gemm_param.d.primary = reinterpret_cast<void *>(d);
  gemm_param.op.tertiary = reinterpret_cast<void *>(&numBatchesVar);
  sgemm.gemm_ext(&gemm_param);
}

extern "C" int64_t plaidml_rt_xsmm_brgemm_offs_relu_dispatch_f32(
    int32_t lda, int32_t ldb, int32_t ldc, int32_t m, int32_t n, int32_t k) {
  libxsmm_blasint lda_int = lda;
  libxsmm_blasint ldb_int = ldb;
  libxsmm_blasint ldc_int = ldc;
  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;

  libxsmm_gemm_shape l_shape;
  libxsmm_bitfield l_flags =
      LIBXSMM_GEMM_FLAGS('N', 'N') | LIBXSMM_GEMM_FLAG_BETA_0;
  libxsmm_bitfield l_prefetch_flags = 0;
  libxsmm_gemm_batch_reduce_config l_brconfig;

  l_shape.m = n_int;
  l_shape.n = m_int;
  l_shape.k = k_int;
  l_shape.lda = ldb_int;
  l_shape.ldb = lda_int;
  l_shape.ldc = ldc_int;
  l_shape.a_in_type = LIBXSMM_DATATYPE_F32;
  l_shape.b_in_type = LIBXSMM_DATATYPE_F32;
  l_shape.out_type = LIBXSMM_DATATYPE_F32;
  l_shape.comp_type = LIBXSMM_DATATYPE_F32;
  l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_OFFSET;
  l_brconfig.br_stride_a_hint = 0;
  l_brconfig.br_stride_b_hint = 0;
  l_brconfig.br_unroll_hint = 0;

  libxsmm_gemm_ext_unary_argops l_argops;
  memset(&l_argops, 0, sizeof(libxsmm_gemm_ext_unary_argops));
  l_argops.ldcp = ldc;
  l_argops.cp_unary_type = LIBXSMM_MELTW_TYPE_UNARY_RELU;

  libxsmm_gemm_ext_binary_postops l_postops;
  memset(&l_postops, 0, sizeof(libxsmm_gemm_ext_binary_postops));
  l_postops.d_in_type = LIBXSMM_DATATYPE_F32;
  l_postops.d_binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
  l_postops.d_binary_type = LIBXSMM_MELTW_TYPE_BINARY_ADD;
  l_postops.ldd = ldc;

  auto sgemm = libxsmm_dispatch_brgemm_ext_v2(
      l_shape, l_flags, l_prefetch_flags, l_brconfig, l_argops, l_postops);

  return reinterpret_cast<int64_t>(sgemm);
}

extern "C" int64_t plaidml_rt_xsmm_brgemm_offs_bias_dispatch_f32(
    int32_t lda, int32_t ldb, int32_t ldc, int32_t m, int32_t n, int32_t k) {
  libxsmm_blasint lda_int = lda;
  libxsmm_blasint ldb_int = ldb;
  libxsmm_blasint ldc_int = ldc;
  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;

  libxsmm_gemm_shape l_shape;
  libxsmm_bitfield l_flags =
      LIBXSMM_GEMM_FLAGS('N', 'N') | LIBXSMM_GEMM_FLAG_BETA_0;
  libxsmm_bitfield l_prefetch_flags = 0;
  libxsmm_gemm_batch_reduce_config l_brconfig;

  l_shape.m = n_int;
  l_shape.n = m_int;
  l_shape.k = k_int;
  l_shape.lda = ldb_int;
  l_shape.ldb = lda_int;
  l_shape.ldc = ldc_int;
  l_shape.a_in_type = LIBXSMM_DATATYPE_F32;
  l_shape.b_in_type = LIBXSMM_DATATYPE_F32;
  l_shape.out_type = LIBXSMM_DATATYPE_F32;
  l_shape.comp_type = LIBXSMM_DATATYPE_F32;
  l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_OFFSET;
  l_brconfig.br_stride_a_hint = 0;
  l_brconfig.br_stride_b_hint = 0;
  l_brconfig.br_unroll_hint = 0;

  libxsmm_gemm_ext_unary_argops l_argops;
  memset(&l_argops, 0, sizeof(libxsmm_gemm_ext_unary_argops));
  l_argops.ldcp = ldc;

  libxsmm_gemm_ext_binary_postops l_postops;
  memset(&l_postops, 0, sizeof(libxsmm_gemm_ext_binary_postops));
  l_postops.d_in_type = LIBXSMM_DATATYPE_F32;
  l_postops.d_binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
  l_postops.d_binary_type = LIBXSMM_MELTW_TYPE_BINARY_ADD;
  l_postops.ldd = ldc;

  auto sgemm = libxsmm_dispatch_brgemm_ext_v2(
      l_shape, l_flags, l_prefetch_flags, l_brconfig, l_argops, l_postops);

  return reinterpret_cast<int64_t>(sgemm);
}

extern "C" int64_t plaidml_rt_xsmm_brgemm_offs_relu_beta1_dispatch_f32(
    int32_t lda, int32_t ldb, int32_t ldc, int32_t m, int32_t n, int32_t k) {
  libxsmm_blasint lda_int = lda;
  libxsmm_blasint ldb_int = ldb;
  libxsmm_blasint ldc_int = ldc;
  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;

  libxsmm_gemm_shape l_shape;
  libxsmm_bitfield l_flags = LIBXSMM_GEMM_FLAGS('N', 'N');
  libxsmm_bitfield l_prefetch_flags = 0;
  libxsmm_gemm_batch_reduce_config l_brconfig;

  l_shape.m = n_int;
  l_shape.n = m_int;
  l_shape.k = k_int;
  l_shape.lda = ldb_int;
  l_shape.ldb = lda_int;
  l_shape.ldc = ldc_int;
  l_shape.a_in_type = LIBXSMM_DATATYPE_F32;
  l_shape.b_in_type = LIBXSMM_DATATYPE_F32;
  l_shape.out_type = LIBXSMM_DATATYPE_F32;
  l_shape.comp_type = LIBXSMM_DATATYPE_F32;
  l_brconfig.br_type = LIBXSMM_GEMM_BATCH_REDUCE_OFFSET;
  l_brconfig.br_stride_a_hint = 0;
  l_brconfig.br_stride_b_hint = 0;
  l_brconfig.br_unroll_hint = 0;

  libxsmm_gemm_ext_unary_argops l_argops;
  memset(&l_argops, 0, sizeof(libxsmm_gemm_ext_unary_argops));
  l_argops.ldcp = ldc;
  l_argops.cp_unary_type = LIBXSMM_MELTW_TYPE_UNARY_RELU;

  libxsmm_gemm_ext_binary_postops l_postops;
  memset(&l_postops, 0, sizeof(libxsmm_gemm_ext_binary_postops));
  l_postops.d_in_type = LIBXSMM_DATATYPE_F32;
  l_postops.d_binary_flags = LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0;
  l_postops.d_binary_type = LIBXSMM_MELTW_TYPE_BINARY_ADD;
  l_postops.ldd = ldc;

  auto sgemm = libxsmm_dispatch_brgemm_ext_v2(
      l_shape, l_flags, l_prefetch_flags, l_brconfig, l_argops, l_postops);

  return reinterpret_cast<int64_t>(sgemm);
}

namespace pmlc::rt {

void registerXsmm() {
  libxsmm_init();

  REGISTER_SYMBOL(plaidml_rt_xsmm_gemm_invoke_f32);
  REGISTER_SYMBOL(plaidml_rt_xsmm_gemm_dispatch_f32);
  REGISTER_SYMBOL(plaidml_rt_xsmm_gemm_relu_invoke_f32);
  REGISTER_SYMBOL(plaidml_rt_xsmm_gemm_relu_dispatch_f32);
  REGISTER_SYMBOL(plaidml_rt_xsmm_gemm_bias_invoke_f32);
  REGISTER_SYMBOL(plaidml_rt_xsmm_gemm_bias_dispatch_f32);
  REGISTER_SYMBOL(plaidml_rt_xsmm_gemm_relu_beta1_invoke_f32);
  REGISTER_SYMBOL(plaidml_rt_xsmm_gemm_relu_beta1_dispatch_f32);
  REGISTER_SYMBOL(plaidml_rt_xsmm_brgemm_invoke_f32);
  REGISTER_SYMBOL(plaidml_rt_xsmm_brgemm_dispatch_f32);
  REGISTER_SYMBOL(plaidml_rt_xsmm_brgemm_relu_invoke_f32);
  REGISTER_SYMBOL(plaidml_rt_xsmm_brgemm_relu_dispatch_f32);
  REGISTER_SYMBOL(plaidml_rt_xsmm_brgemm_bias_invoke_f32);
  REGISTER_SYMBOL(plaidml_rt_xsmm_brgemm_bias_dispatch_f32);
  REGISTER_SYMBOL(plaidml_rt_xsmm_brgemm_relu_beta1_invoke_f32);
  REGISTER_SYMBOL(plaidml_rt_xsmm_brgemm_relu_beta1_dispatch_f32);
  REGISTER_SYMBOL(plaidml_rt_xsmm_brgemm_offs_invoke_f32);
  REGISTER_SYMBOL(plaidml_rt_xsmm_brgemm_offs_dispatch_f32);
  REGISTER_SYMBOL(plaidml_rt_xsmm_brgemm_offs_relu_invoke_f32);
  REGISTER_SYMBOL(plaidml_rt_xsmm_brgemm_offs_relu_dispatch_f32);
  REGISTER_SYMBOL(plaidml_rt_xsmm_brgemm_offs_bias_invoke_f32);
  REGISTER_SYMBOL(plaidml_rt_xsmm_brgemm_offs_bias_dispatch_f32);
  REGISTER_SYMBOL(plaidml_rt_xsmm_brgemm_offs_relu_beta1_invoke_f32);
  REGISTER_SYMBOL(plaidml_rt_xsmm_brgemm_offs_relu_beta1_dispatch_f32);
  REGISTER_SYMBOL(plaidml_rt_xsmm_unary_dispatch);
  REGISTER_SYMBOL(plaidml_rt_xsmm_unary_invoke);
  REGISTER_SYMBOL(plaidml_rt_xsmm_binary_dispatch);
  REGISTER_SYMBOL(plaidml_rt_xsmm_binary_invoke);
}

} // namespace pmlc::rt
