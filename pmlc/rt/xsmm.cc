// Copyright 2020 Intel Corporation

#ifndef _WIN32
#include "libxsmm_source.h" // NOLINT [build/include_subdir]
#endif

#include "pmlc/rt/memref.h"

#ifndef _WIN32
struct Initializer {
  Initializer() { libxsmm_init(); }
};

static Initializer init;
#endif

#ifndef _WIN32
extern "C" void plaidml_rt_xsmm_gemm_f32(            //
    size_t aRank, StridedMemRefType<float, 2> *aRef, //
    size_t bRank, StridedMemRefType<float, 2> *bRef, //
    size_t cRank, StridedMemRefType<float, 2> *cRef, //
    int32_t lda, int32_t ldb, int32_t ldc,           //
    int32_t m, int32_t n, int32_t k) {
  auto aPtr = aRef->data + aRef->offset;
  auto bPtr = bRef->data + bRef->offset;
  auto cPtr = cRef->data + cRef->offset;

  libxsmm_blasint lda_int = lda;
  libxsmm_blasint ldb_int = ldb;
  libxsmm_blasint ldc_int = ldc;
  // auto sgemm = libxsmm_smmdispatch(m, n, k, &lda_int, &ldb_int, &ldc_int,
  //                                  /*alpha=*/nullptr,
  //                                  /*beta=*/nullptr, /*flags=*/nullptr,
  //                                  /*prefetch=*/nullptr);
  // sgemm(aPtr, bPtr, cPtr);
  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;
  libxsmm_gemm(/*transa=*/nullptr, /*transb=*/nullptr, &m_int, &n_int, &k_int,
               /*alpha=*/nullptr, bPtr, &ldb_int, aPtr, &lda_int,
               /*beta=*/nullptr, cPtr, &ldc_int);
#endif
}
