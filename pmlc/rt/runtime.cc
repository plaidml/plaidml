// Copyright 2020 Intel Corporation

#include <cmath>
#include <cstdlib>

#include "llvm/Support/raw_ostream.h"

#include "mlir/ExecutionEngine/RunnerUtils.h"

#include "pmlc/compiler/registry.h"

extern "C" void plaidml_rt_trace(const char *msg) {
  llvm::outs() << msg << "\n";
  llvm::outs().flush();
}

namespace {
struct Registration {
  Registration() {
    using pmlc::compiler::registerSymbol;

    // cstdlib functions
    registerSymbol("free", reinterpret_cast<void *>(free));
    registerSymbol("malloc", reinterpret_cast<void *>(malloc));

    // cmath functions
    registerSymbol("ceilf", reinterpret_cast<void *>(ceilf));
    registerSymbol("expf", reinterpret_cast<void *>(expf));
    registerSymbol("logf", reinterpret_cast<void *>(logf));

    // RunnerUtils functions
    registerSymbol("print_memref_f32",
                   reinterpret_cast<void *>(print_memref_f32));

    registerSymbol("plaidml_rt_trace",
                   reinterpret_cast<void *>(plaidml_rt_trace));
  }
};
static Registration reg;
} // namespace
