// Copyright 2020 Intel Corporation

#include "mlir/ExecutionEngine/RunnerUtils.h"

#include "pmlc/compiler/registry.h"
#include "pmlc/util/logging.h"

extern "C" void
_mlir_ciface_plaidml_rt_prng(UnrankedMemRefType<uint32_t> *state,
                             UnrankedMemRefType<float> *result,
                             UnrankedMemRefType<uint32_t> *newState) {
  DynamicMemRefType<uint32_t> stateRef(*state);
  DynamicMemRefType<float> resultRef(*result);
  DynamicMemRefType<uint32_t> newStateRef(*newState);

  unsigned count = 1;
  for (unsigned i = 0; i < resultRef.rank; i++) {
    count *= resultRef.sizes[i];
  }

  uint32_t *statePtr = stateRef.data + stateRef.offset;
  float *resultPtr = resultRef.data + resultRef.offset;
  uint32_t *newStatePtr = newStateRef.data + newStateRef.offset;

  uint32_t s0 = statePtr[0];
  uint32_t s1 = statePtr[1];
  uint32_t s2 = statePtr[2];

  for (unsigned i = 0; i < count; ++i) {
    resultPtr[i] = (s0 ^ s1 ^ s2) / 4294967296.0;
    s0 = (((s0 & 4294967294) << 12) ^ (((s0 << 13) ^ s0) >> 19));
    s1 = (((s1 & 4294967288) << 4) ^ (((s1 << 2) ^ s1) >> 25));
    s2 = (((s2 & 4294967280) << 17) ^ (((s2 << 3) ^ s2) >> 11));
  }

  newStatePtr[0] = s0;
  newStatePtr[1] = s1;
  newStatePtr[2] = s2;
}

namespace {
struct Registration {
  Registration() {
    using pmlc::compiler::registerSymbol;
    registerSymbol("_mlir_ciface_plaidml_rt_prng",
                   reinterpret_cast<void *>(_mlir_ciface_plaidml_rt_prng));
  }
};
static Registration reg;
} // namespace
