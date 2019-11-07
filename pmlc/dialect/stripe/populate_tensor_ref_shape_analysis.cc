// Copyright 2019 Intel Corporation

#include <algorithm>

#include "pmlc/dialect/stripe/analysis.h"
#include "pmlc/dialect/stripe/populate_tensor_ref_shape_analysis.h"

#include "mlir/Pass/Pass.h"

#define PASS_NAME "test-populate-tensor-ref-shape"
#define DEBUG_TYPE PASS_NAME

using llvm::cast;
using llvm::isa;
using mlir::BlockArgument;
using mlir::FuncOp;
using mlir::FunctionType;
using mlir::SmallVector;

namespace pmlc {
namespace dialect {
namespace stripe {

PopulateTensorRefShape::PopulateTensorRefShape(mlir::Operation* op) : operation(op) {
  assert(llvm::isa<FuncOp>(op) && "Only FuncOp is supported in PopulateTensorRefShape");
}

// Retrieve the layout information for each Value in `valueList` with a 'tensor_ref' type and replace the type with a
// 'tensor_ref' that contains the shape information.
template <class Range>
static void populateValuesWithShapes(Range valueRange) {
  for (Value* result : valueRange) {
    if (result->getType().isa<TensorRefType>()) {
      result->setType(
          TensorRefType::get(pmlc::dialect::stripe::ComputeAccess(result).base_type, /*propagateShape=*/true));
    }
  }
}

void PopulateTensorRefShape::recompute() {
  FuncOp func = llvm::cast<FuncOp>(operation);

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

namespace {

class TestPopulateTensorRefShape : public mlir::FunctionPass<TestPopulateTensorRefShape> {
 public:
  void runOnFunction() override {
    pmlc::dialect::stripe::PopulateTensorRefShape analysis(getFunction());
    analysis.recompute();
  }
};
}  // namespace

std::unique_ptr<mlir::Pass> mlir::createTestPopulateTensorRefShapePass() {
  return std::make_unique<TestPopulateTensorRefShape>();
}

static mlir::PassRegistration<TestPopulateTensorRefShape> pass(
    PASS_NAME, "Pass for testing analysis that populates 'tensor_ref' types with shape information.");
