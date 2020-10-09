// Copyright 2020 Intel Corporation

#pragma once

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/InterfaceSupport.h"
#include "mlir/Support/LogicalResult.h"

namespace pmlc::util {

using mlir::failure;
using mlir::LogicalResult;
using mlir::OpBuilder;

#include "pmlc/util/interfaces.h.inc"

} // namespace pmlc::util
