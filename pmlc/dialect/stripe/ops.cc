// Copyright 2019, Intel Corporation

#include "pmlc/dialect/stripe/ops.h"
#include "mlir/IR/OpImplementation.h"

namespace pmlc {
namespace dialect {
namespace stripe {

#define GET_OP_CLASSES
#include "pmlc/dialect/stripe/ops.cpp.inc"

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc

#include "pmlc/dialect/stripe/ops_enum.cpp.inc"
#include "llvm/ADT/StringSwitch.h"
