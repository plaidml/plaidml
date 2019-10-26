// Copyright 2019 Intel Corporation

#include <algorithm>

#include "pmlc/dialect/stripe/analysis.h"
#include "pmlc/dialect/stripe/dialect.h"
#include "pmlc/dialect/stripe/populate_tensor_ref_shape_pass.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define PASS_NAME "populate-tensor-ref-shape"
#define DEBUG_TYPE PASS_NAME

using mlir::BlockArgument;
using mlir::FuncOp;
using mlir::FunctionType;
using mlir::Operation;
using mlir::SmallVector;
using mlir::Type;
using mlir::Value;
using pmlc::dialect::stripe::TensorRefType;

namespace {

/// This pass populates the optional shape information of 'tensor_ref' types. This information is found in the 'layout'
/// attribute of 'tensor_ref' types that are used in BlockArgument values and Alloc operations. For each 'tensor_ref'
/// Value, the algorithm proceed as follows:
///   1. Traverse the use-def chain until a BlockArgument/Alloc is found and retrieve its 'layout' attribute.
///   2. Replace Value's 'tensor_ref' type with a new 'tensor_ref' type that contains the shape information.
// NOTE: This analysis is O(n^2) so we should revisit this approach/design if the layout information is needed in
// more cases than the Stripe->Affine dialect conversion.
class PopulateTensorRefShape : public mlir::FunctionPass<PopulateTensorRefShape> {
 public:
  PopulateTensorRefShape() = default;

  /// Constructor to use this pass as analysis.
  explicit PopulateTensorRefShape(mlir::Operation* op);

  void runOnFunction() override;

 private:
  void recompute(mlir::FuncOp func);
};

// Retrieve the layout information for each Value in `valueList` with a 'tensor_ref' type and replace the type with a
// 'tensor_ref' that contains the shape information.
template <class Range>
static void visitValues(Range valueRange) {
  for (Value* result : valueRange) {
    if (result->getType().isa<TensorRefType>()) {
      result->setType(TensorRefType::get(pmlc::dialect::stripe::ComputeAccess(result).base_type));
    }
  }
}

void PopulateTensorRefShape::recompute(FuncOp func) {
  // Visit types in block arguments.
  for (auto& block : func.getBody().getBlocks()) {
    visitValues(block.getArguments());
  }

  // Visit function type. Propagate types from entry block arguments to function type. We don't expect TensorRef types
  // in function return.
  auto funcArgs = func.getArguments();
  SmallVector<Type, 8> newInputs;
  std::transform(funcArgs.begin(), funcArgs.end(), std::back_inserter(newInputs),
                 [](const BlockArgument* arg) { return arg->getType(); });
  func.setType(FunctionType::get(newInputs, func.getType().getResults(), &getContext()));

  // Visit types in nested operations.
  func.walk([](Operation* op) { visitValues(op->getResults()); });
}

void PopulateTensorRefShape::runOnFunction() {
  recompute(getFunction());
  // TODO: Line below breaks linking.
  // markAllAnalysesPreserved();
}

}  // namespace

std::unique_ptr<mlir::FunctionPassBase> mlir::createPopulateTensorRefShapePass() {
  return std::make_unique<PopulateTensorRefShape>();
}

static mlir::PassRegistration<PopulateTensorRefShape> pass(PASS_NAME,
                                                           "Populate 'tensor_ref' types with shape information.");
