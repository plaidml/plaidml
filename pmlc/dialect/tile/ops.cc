// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/ops.h"

#include "mlir/IR/OpImplementation.h"

namespace pmlc {
namespace dialect {
namespace tile {

#define GET_OP_CLASSES
#include "pmlc/dialect/tile/ops.cpp.inc"

}  // namespace tile
}  // namespace dialect
}  // namespace pmlc
