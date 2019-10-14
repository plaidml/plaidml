// Copyright 2019, Intel Corporation

#pragma once

#include "llvm/ADT/StringRef.h"

#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/StandardTypes.h"

namespace pmlc {
namespace util {

// Adjust the result types on the containing FuncOp if this op relates to an output
void UpdateFuncOpType(mlir::Operation* op);

llvm::StringRef getOpName(const mlir::OperationName& name);

}  // namespace util
}  // namespace pmlc
