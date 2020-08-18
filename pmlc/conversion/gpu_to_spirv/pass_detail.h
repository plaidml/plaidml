// Copyright 2020, Intel Corporation

#pragma once

#include "mlir/Pass/Pass.h"

namespace pmlc::conversion::gpu_to_spirv {

#define GEN_PASS_CLASSES
#include "pmlc/conversion/gpu_to_spirv/passes.h.inc"

} // namespace pmlc::conversion::gpu_to_spirv
