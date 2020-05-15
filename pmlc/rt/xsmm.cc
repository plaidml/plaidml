// Copyright 2020 Intel Corporation

#include "mlir/ExecutionEngine/RunnerUtils.h"

#include "libxsmm.h" // NOLINT [build/include_subdir]

#include "pmlc/compiler/registry.h"
#include "pmlc/util/logging.h"

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
  libxsmm_blasint m_int = m;
  libxsmm_blasint n_int = n;
  libxsmm_blasint k_int = k;

  auto sgemm =
      libxsmm_smmdispatch(n_int, m_int, k_int, &ldb_int, &lda_int, &ldc_int,
                          /*alpha=*/nullptr, /*beta=*/nullptr,
                          /*flags=*/nullptr, /*prefetch=*/nullptr);
  sgemm(bPtr, aPtr, cPtr);
}

namespace {
struct Registration {
  Registration() {
    libxsmm_init();

    using pmlc::compiler::registerSymbol;
    registerSymbol("plaidml_rt_xsmm_gemm_f32",
                   reinterpret_cast<void *>(plaidml_rt_xsmm_gemm_f32));
  }
};
static Registration reg;
} // namespace
