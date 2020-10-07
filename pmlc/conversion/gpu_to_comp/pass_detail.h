// Copyright 2020, Intel Corporation
#pragma once

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Pass/Pass.h"

#include "pmlc/dialect/comp/ir/dialect.h"

namespace pmlc::conversion::gpu_to_comp {

#define GEN_PASS_CLASSES
#include "pmlc/conversion/gpu_to_comp/passes.h.inc"

} // namespace pmlc::conversion::gpu_to_comp
