// Copyright 2019, Intel Corporation

#include "pmlc/dialect/hir/ops.h"

#include "mlir/IR/OpImplementation.h"

namespace pmlc {
namespace dialect {
namespace hir {

#define GET_OP_CLASSES
#include "pmlc/dialect/hir/ops.cpp.inc"

}  // namespace hir
}  // namespace dialect
}  // namespace pmlc
