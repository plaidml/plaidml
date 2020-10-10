// Copyright 2020 Intel Corporation

#include "openmp/runtime/src/kmp.h"

#include "pmlc/rt/symbol_registry.h"

namespace {
struct Registration {
  Registration() {
    using pmlc::rt::registerSymbol;

    registerSymbol("__kmpc_fork_call",
                   reinterpret_cast<void *>(__kmpc_fork_call));

    registerSymbol("__kmpc_global_thread_num",
                   reinterpret_cast<void *>(__kmpc_global_thread_num));
  }
};
static Registration reg;
} // namespace
