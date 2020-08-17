// Copyright 2020 Intel Corporation

#include "mlir/ExecutionEngine/RunnerUtils.h"

#include "libxsmm.h" // NOLINT [build/include_subdir]

#include "pmlc/compiler/registry.h"
#include "pmlc/util/logging.h"

using FunctionPtr = void (*)(const void *, const void *, void *, ...);

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

namespace {
struct Registration {
  Registration() {
    libxsmm_init();

    using pmlc::compiler::registerSymbol;

    registerSymbol("plaidml_rt_xsmm_gemm_invoke_f32",
                   reinterpret_cast<void *>(plaidml_rt_xsmm_gemm_invoke_f32));

    registerSymbol("plaidml_rt_xsmm_gemm_dispatch_f32",
                   reinterpret_cast<void *>(plaidml_rt_xsmm_gemm_dispatch_f32));
  }
};
static Registration reg;
} // namespace
