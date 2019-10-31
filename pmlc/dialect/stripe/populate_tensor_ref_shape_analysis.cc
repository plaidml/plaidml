// Copyright 2019 Intel Corporation

#include <algorithm>

#include "pmlc/dialect/stripe/analysis.h"
#include "pmlc/dialect/stripe/populate_tensor_ref_shape_analysis.h"

using mlir::BlockArgument;
using mlir::FuncOp;
using mlir::FunctionType;
using mlir::SmallVector;

namespace pmlc {
namespace dialect {
namespace stripe {

PopulateTensorRefShape::PopulateTensorRefShape(mlir::Operation* op) {}

// Retrieve the layout information for each Value in `valueList` with a 'tensor_ref' type and replace the type with a
// 'tensor_ref' that contains the shape information.
template <class Range>
static void populateValuesWithShapes(Range valueRange) {
  for (Value* result : valueRange) {
    if (result->getType().isa<TensorRefType>()) {
      result->setType(TensorRefType::get(pmlc::dialect::stripe::ComputeAccess(result).base_type));
    }
  }
}

void PopulateTensorRefShape::populateWithShapes(FuncOp func) {
  // Visit types in block arguments.
  for (auto& block : func.getBody().getBlocks()) {
    populateValuesWithShapes(block.getArguments());
  }

  // Visit function type. Propagate types from entry block arguments to function type. We don't expect TensorRef types
  // in function return.
  auto funcArgs = func.getArguments();
  SmallVector<Type, 8> newInputs;
  std::transform(funcArgs.begin(), funcArgs.end(), std::back_inserter(newInputs),
                 [](const BlockArgument* arg) { return arg->getType(); });
  func.setType(FunctionType::get(newInputs, func.getType().getResults(), func.getContext()));

  // Visit types in nested operations.
  func.walk([](Operation* op) { populateValuesWithShapes(op->getResults()); });
}

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
