// Copyright 2019, Intel Corporation

#include "pmlc/dialect/mir/ops.h"
#include "mlir/IR/OpImplementation.h"

namespace pmlc {
namespace dialect {
namespace mir {

#define GET_OP_CLASSES
#include "pmlc/dialect/mir/ops.cpp.inc"

}  // namespace mir
}  // namespace dialect
}  // namespace pmlc
