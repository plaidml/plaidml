// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/types.h"

namespace pmlc::dialect::tile {

AffineTensorMapType AffineTensorMapType::get(mlir::MLIRContext* context) {  //
  return Base::get(context, Kinds::AffineTensorMap);
}

AffineMapType AffineMapType::get(mlir::MLIRContext* context) {  //
  return Base::get(context, Kinds::AffineMap);
}

AffineIndexMapType AffineIndexMapType::get(mlir::MLIRContext* context) {  //
  return Base::get(context, Kinds::AffineIndexMap);
}

AffineSizeMapType AffineSizeMapType::get(mlir::MLIRContext* context) {  //
  return Base::get(context, Kinds::AffineSizeMap);
}

AffineConstraintsType AffineConstraintsType::get(mlir::MLIRContext* context) {  //
  return Base::get(context, Kinds::AffineConstraints);
}

StringType StringType::get(mlir::MLIRContext* context) {  //
  return Base::get(context, Kinds::String);
}

}  // namespace pmlc::dialect::tile
