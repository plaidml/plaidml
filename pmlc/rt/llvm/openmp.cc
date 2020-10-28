// Copyright 2020 Intel Corporation

#include <kmp.h>

#include "pmlc/rt/symbol_registry.h"

extern "C" uint64_t _mlir_ciface_plaidml_rt_thread_num() {
  return __kmpc_bound_thread_num(nullptr);
}

namespace {

struct Registration {
  Registration() {
    using pmlc::rt::registerSymbol;

    registerSymbol("__kmpc_barrier", reinterpret_cast<void *>(__kmpc_barrier));
    registerSymbol("__kmpc_flush", reinterpret_cast<void *>(__kmpc_flush));
    registerSymbol("__kmpc_fork_call",
                   reinterpret_cast<void *>(__kmpc_fork_call));
    registerSymbol("__kmpc_global_thread_num",
                   reinterpret_cast<void *>(__kmpc_global_thread_num));
    registerSymbol("__kmpc_omp_taskwait",
                   reinterpret_cast<void *>(__kmpc_omp_taskwait));
    registerSymbol("__kmpc_omp_taskyield",
                   reinterpret_cast<void *>(__kmpc_omp_taskyield));
    registerSymbol("__kmpc_push_num_threads",
                   reinterpret_cast<void *>(__kmpc_push_num_threads));
    registerSymbol(
        "_mlir_ciface_plaidml_rt_thread_num",
        reinterpret_cast<void *>(_mlir_ciface_plaidml_rt_thread_num));
  }
};

static Registration reg;

} // namespace
