// Copyright 2020, Intel Corporation
#pragma once

#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Pass/Pass.h"

namespace pmlc::target::intel_gen_ocl_spirv {

#define GEN_PASS_CLASSES
#include "pmlc/target/intel_gen_ocl_spirv/passes.h.inc"

} // namespace pmlc::target::intel_gen_ocl_spirv
