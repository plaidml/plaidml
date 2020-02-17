// Copyright 2020 Intel Corporation

#include "pmlc/dialect/xsmm/ir/dialect.h"

#include "pmlc/dialect/xsmm/ir/ops.h"

namespace pmlc::dialect::xsmm {

Dialect::Dialect(mlir::MLIRContext *ctx)
    : mlir::Dialect(getDialectNamespace(), ctx) {
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/xsmm/ir/ops.cc.inc"
      >();
}

} // namespace pmlc::dialect::xsmm
