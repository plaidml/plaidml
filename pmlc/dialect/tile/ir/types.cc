// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/ir/types.h"

namespace pmlc::dialect::tile {

AffineMapType AffineMapType::get(mlir::MLIRContext *context) {
  return Base::get(context);
}

AffineTensorMapType AffineTensorMapType::get(mlir::MLIRContext *context) {
  return Base::get(context);
}

AffineConstraintsType AffineConstraintsType::get(mlir::MLIRContext *context) {
  return Base::get(context);
}

StringType StringType::get(mlir::MLIRContext *context) {
  return Base::get(context);
}

} // namespace pmlc::dialect::tile
