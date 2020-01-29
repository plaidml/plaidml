// Copyright 2020 Intel Corporation

#include "pmlc/dialect/stdx/ir/dialect.h"

#include "pmlc/dialect/stdx/ir/ops.h"

namespace pmlc::dialect::stdx {

Dialect::Dialect(mlir::MLIRContext* ctx) : mlir::Dialect(getDialectNamespace(), ctx) {
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/stdx/ir/ops.cc.inc"
      >();
}

llvm::StringRef Dialect::getDialectNamespace() { return "stdx"; }

}  // namespace pmlc::dialect::stdx
