// Copyright 2020 Intel Corporation

#include "openmp/runtime/src/kmp.h"

#include "pmlc/rt/symbol_registry.h"

extern "C" uint64_t _mlir_ciface_plaidml_rt_thread_num() {
  uint64_t tid = __kmp_get_tid();
  return tid;
}

namespace {
struct Registration {
  Registration() {
    using pmlc::rt::registerSymbol;

    registerSymbol(
        "_mlir_ciface_plaidml_rt_thread_num",
        reinterpret_cast<void *>(_mlir_ciface_plaidml_rt_thread_num));
  }
};

static Registration reg;
} // namespace
