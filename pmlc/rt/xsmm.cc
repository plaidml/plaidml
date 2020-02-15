// Copyright 2020 Intel Corporation

#include "libxsmm_source.h" // NOLINT [build/include_subdir]

#include "pmlc/rt/memref.h"

extern "C" void plaidml_rt_xsmm_gemm_f32(UnrankedMemRefType a,
                                         UnrankedMemRefType b,
                                         UnrankedMemRefType c, int32_t lda,
                                         int32_t ldb, int32_t ldc, int32_t m,
                                         int32_t n, int32_t k) {
  auto aRef = static_cast<StridedMemRefType<float, 2> *>(a.descriptor);
  auto bRef = static_cast<StridedMemRefType<float, 2> *>(b.descriptor);
  auto cRef = static_cast<StridedMemRefType<float, 2> *>(c.descriptor);

  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;
  libxsmm_blasint lda_int = lda;
  libxsmm_blasint ldb_int = ldb;
  libxsmm_blasint ldc_int = ldc;
  // auto sgemm = libxsmm_smmdispatch(m, n, k, &lda_int, &ldb_int, &ldc_int,
  //                                  /*alpha=*/nullptr,
  //                                  /*beta=*/nullptr, /*flags=*/nullptr,
  //                                  /*prefetch=*/nullptr);
  auto aPtr = aRef->data + aRef->offset;
  auto bPtr = bRef->data + bRef->offset;
  auto cPtr = cRef->data + cRef->offset;
  // sgemm(aPtr, bPtr, cPtr);
  libxsmm_gemm(/*transa=*/nullptr, /*transb=*/nullptr, &m_int, &n_int, &k_int,
               /*alpha=*/nullptr, aPtr, &lda_int, bPtr, &ldb_int,
               /*beta=*/nullptr, cPtr, &ldc_int);
}
