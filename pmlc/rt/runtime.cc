// Copyright 2020 Intel Corporation

#include <cmath>
#include <cstdlib>

#include "half.hpp"
#include "llvm/Support/raw_ostream.h"

#include "mlir/ExecutionEngine/RunnerUtils.h"

#include "pmlc/compiler/registry.h"

extern "C" void plaidml_rt_trace(const char *msg) {
  llvm::outs() << msg << "\n";
  llvm::outs().flush();
}

float h2f(half_float::half n) { return n; }

half_float::half f2h(float n) {
  return half_float::half_cast<half_float::half>(n);
}

namespace {
struct Registration {
  Registration() {
    using pmlc::compiler::registerSymbol;

    // compiler_rt functions
    // TODO: replace with @llvm-project//compiler-rt/lib/builtins
    registerSymbol("__gnu_h2f_ieee", reinterpret_cast<void *>(h2f));
    registerSymbol("__gnu_f2h_ieee", reinterpret_cast<void *>(f2h));
    registerSymbol("___extendhfsf2", reinterpret_cast<void *>(h2f));
    registerSymbol("___truncsfhf2", reinterpret_cast<void *>(f2h));

    // cstdlib functions
    registerSymbol("free", reinterpret_cast<void *>(free));
    registerSymbol("malloc", reinterpret_cast<void *>(malloc));

    // cmath functions
    registerSymbol("_mlir_ciface_acosf", reinterpret_cast<void *>(acosf));
    registerSymbol("_mlir_ciface_asinf", reinterpret_cast<void *>(asinf));
    registerSymbol("_mlir_ciface_atanf", reinterpret_cast<void *>(atanf));
    registerSymbol("_mlir_ciface_ceilf", reinterpret_cast<void *>(ceilf));
    registerSymbol("_mlir_ciface_coshf", reinterpret_cast<void *>(coshf));
    registerSymbol("_mlir_ciface_erff", reinterpret_cast<void *>(erff));
    registerSymbol("_mlir_ciface_expf", reinterpret_cast<void *>(expf));
    registerSymbol("_mlir_ciface_floorf", reinterpret_cast<void *>(floorf));
    registerSymbol("_mlir_ciface_logf", reinterpret_cast<void *>(logf));
    registerSymbol("_mlir_ciface_powf", reinterpret_cast<void *>(powf));
    registerSymbol("_mlir_ciface_roundf", reinterpret_cast<void *>(roundf));
    registerSymbol("_mlir_ciface_sinhf", reinterpret_cast<void *>(sinhf));
    registerSymbol("_mlir_ciface_tanf", reinterpret_cast<void *>(tanf));
    registerSymbol("_mlir_ciface_tanhf", reinterpret_cast<void *>(tanhf));

    // RunnerUtils functions
    registerSymbol("_mlir_ciface_print_memref_f32",
                   reinterpret_cast<void *>(_mlir_ciface_print_memref_f32));

    registerSymbol("plaidml_rt_trace",
                   reinterpret_cast<void *>(plaidml_rt_trace));
  }
};
static Registration reg;
} // namespace
