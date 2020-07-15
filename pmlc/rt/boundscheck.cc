// Copyright 2020 Intel Corporation

#include <cstdint>
#include <stdexcept>

#include "pmlc/compiler/registry.h"

// This function is called from the BoundsCheckPass.
// index - index for a dimension that is accessed by load/store operation
// range - upper bound of the range for this dimension (it is always 0 -
// range)
extern "C" void _mlir_ciface_plaidml_rt_bounds_check(intptr_t index,
                                                     int64_t range) {
  int64_t accessIndex = static_cast<int64_t>(index);
  if (accessIndex < 0 || accessIndex >= range)
    throw std::runtime_error(
        "Out of bounds index for mlir::LoadOp or mlir::StoreOp");
}

namespace {
struct Registration {
  Registration() {
    using pmlc::compiler::registerSymbol;
    registerSymbol(
        "_mlir_ciface_plaidml_rt_bounds_check",
        reinterpret_cast<void *>(_mlir_ciface_plaidml_rt_bounds_check));
  }
};
static Registration reg;
} // namespace
