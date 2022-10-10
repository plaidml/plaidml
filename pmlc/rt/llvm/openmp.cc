// Copyright 2020 Intel Corporation

#include <kmp.h>

#include "pmlc/rt/symbol_registry.h"

extern "C" void __kmpc_for_static_init_8u(void *, int32_t, int32_t, int32_t *,
                                          uint64_t *, uint64_t *, int64_t *,
                                          int64_t, int64_t);

extern "C" uint64_t _mlir_ciface_plaidml_rt_thread_num() {
  return __kmpc_bound_thread_num(nullptr);
}

namespace pmlc::rt::llvm {

void registerOpenMP() {
  REGISTER_SYMBOL(__kmpc_barrier);
  REGISTER_SYMBOL(__kmpc_flush);
  REGISTER_SYMBOL(__kmpc_for_static_init_8u);
  REGISTER_SYMBOL(__kmpc_dispatch_next_8u);
  REGISTER_SYMBOL(__kmpc_dispatch_init_8u);
  REGISTER_SYMBOL(__kmpc_for_static_fini);
  REGISTER_SYMBOL(__kmpc_fork_call);
  REGISTER_SYMBOL(__kmpc_global_thread_num);
  REGISTER_SYMBOL(__kmpc_omp_taskwait);
  REGISTER_SYMBOL(__kmpc_omp_taskyield);
  REGISTER_SYMBOL(__kmpc_push_num_threads);
  REGISTER_SYMBOL(_mlir_ciface_plaidml_rt_thread_num);
}

} // namespace pmlc::rt::llvm
