// Copyright 2020, Intel Corporation

#include "pmlc/dialect/pxa/ir/dialect.h"

#include "pmlc/dialect/pxa/ir/ops.h"

namespace pmlc::dialect::pxa {

Dialect::Dialect(mlir::MLIRContext *ctx)
    : mlir::Dialect(getDialectNamespace(), ctx) {
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/pxa/ir/ops.cc.inc"
      >();
}

llvm::StringRef Dialect::getDialectNamespace() { return "pxa"; }

} // namespace pmlc::dialect::pxa
