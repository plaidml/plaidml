// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/ir/types.h"

namespace pmlc::dialect::tile {

AffineMapType AffineMapType::get(mlir::MLIRContext *context) {
  return Base::get(context, TypeKinds::AffineMap);
}

AffineTensorMapType AffineTensorMapType::get(mlir::MLIRContext *context) {
  return Base::get(context, TypeKinds::AffineTensorMap);
}

AffineConstraintsType AffineConstraintsType::get(mlir::MLIRContext *context) {
  return Base::get(context, TypeKinds::AffineConstraints);
}

StringType StringType::get(mlir::MLIRContext *context) {
  return Base::get(context, TypeKinds::String);
}

} // namespace pmlc::dialect::tile
