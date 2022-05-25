// Copyright 2020, Intel Corporation

#include "pmlc/dialect/layer/ir/ops.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"

#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::layer {

using llvm::SmallVector;

Block *BoxOp::getBody() { return &body().front(); }

void BoxOp::build(OpBuilder &builder, OperationState &result, StringRef op,
                  ValueRange operands, TypeRange resultTypes,
                  DictionaryAttr attrs) {
  result.addTypes(resultTypes);
  result.addOperands(operands);
  result.addAttribute("op", builder.getStringAttr(op));
  result.addAttribute("attrs", attrs);
  // Create a region and a block for the body.
  Region *bodyRegion = result.addRegion();
  auto *body = new Block();
  // Add all the block arguments.
  for (Value operand : operands) {
    body->addArgument(operand.getType(), operand.getLoc());
  }
  bodyRegion->push_back(body);
}

void LayerDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/layer/ir/ops.cc.inc"
      >();
}

} // namespace pmlc::dialect::layer

#include "pmlc/dialect/layer/ir/dialect.cc.inc" // NOLINT

#define GET_OP_CLASSES
#include "pmlc/dialect/layer/ir/ops.cc.inc" // NOLINT
