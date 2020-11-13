// Copyright 2020 Intel Corporation
#pragma once

#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "pmlc/dialect/abi/ir/dialect.h"
#include "pmlc/util/ids.h"

namespace pmlc::dialect::abi {

#define GEN_PASS_CLASSES
#include "pmlc/dialect/abi/transforms/passes.h.inc"

} // namespace pmlc::dialect::abi
