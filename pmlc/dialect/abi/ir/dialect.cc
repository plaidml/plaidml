// Copyright 2020, Intel Corporation

#include "pmlc/dialect/abi/ir/dialect.h"

namespace pmlc::dialect::abi {

using namespace mlir; // NOLINT

#define GET_OP_CLASSES
#include "pmlc/dialect/abi/ir/ops.cc.inc"

void ABIDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/abi/ir/ops.cc.inc" // NOLINT
      >();
}

} // namespace pmlc::dialect::abi
