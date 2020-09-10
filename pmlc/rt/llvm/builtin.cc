// Copyright 2020 Intel Corporation

#include <cmath>

#include "pmlc/rt/symbol_registry.h"

namespace {
struct Registration {
  Registration() {
    using pmlc::rt::registerSymbol;

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
  }
};

static Registration reg;
} // namespace
