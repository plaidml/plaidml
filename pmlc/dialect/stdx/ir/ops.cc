// Copyright 2020 Intel Corporation

#include "pmlc/dialect/stdx/ir/ops.h"

#include "mlir/IR/OpImplementation.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::stdx {

// ---- ReshapeOp ----

Value ReshapeOp::getViewSource() { return tensor(); }

// ---- SubgroupBlockReadINTELOp ----

void SubgroupBlockReadINTELOp::build(OpBuilder &builder, OperationState &result,
                                     Value memref, ValueRange indices) {
  auto memrefType = memref.getType().cast<MemRefType>();
  result.addOperands(memref);
  result.addOperands(indices);
  result.types.push_back(memrefType.getElementType());
}

static LogicalResult
verifySubgroupBlockReadINTELOp(SubgroupBlockReadINTELOp op) {
  if (op.getNumOperands() != 1 + op.getMemRefType().getRank())
    return op.emitOpError("incorrect number of indices for memory ref");
  return success();
}

// ---- SubgroupBlockWriteINTELOp ----

void SubgroupBlockWriteINTELOp::build(OpBuilder &builder,
                                      OperationState &result,
                                      Value valueToStore, Value memref) {
  result.addOperands(valueToStore);
  result.addOperands(memref);
}

static LogicalResult
verifySubgroupBlockWriteINTELOp(SubgroupBlockWriteINTELOp op) {
  if (op.getNumOperands() != 2 + op.getMemRefType().getRank())
    return op.emitOpError("subgroup block write index operand"
                          " count not equal to memref rank");

  return success();
}

static LogicalResult verifyPackOp(PackOp op) {
  auto tupleType = op.getTupleType();
  if (op.getNumOperands() != tupleType.size()) {
    return failure();
  }
  for (unsigned i = 0; i < tupleType.size(); i++) {
    if (op.getOperand(i).getType() != tupleType.getType(i)) {
      return failure();
    }
  }
  return success();
}

static LogicalResult verifyUnpackOp(UnpackOp op) {
  auto tupleType = op.getTupleType();
  if (op.getNumResults() != tupleType.size()) {
    return failure();
  }
  for (unsigned i = 0; i < tupleType.size(); i++) {
    if (op.getResult(i).getType() != tupleType.getType(i)) {
      return failure();
    }
  }
  return success();
}

static ParseResult parseTupleTypeAsm(OpAsmParser &parser, Type &tuple,
                                     SmallVectorImpl<Type> &expanded) {
  auto ret = parser.parseTypeList(expanded);
  if (ret) {
    return failure();
  }
  tuple = TupleType::get(expanded, parser.getBuilder().getContext());
  return ret;
}

static void printTupleTypeAsm(OpAsmPrinter &printer, Type tuple,
                              TypeRange expanded) {
  printer << expanded;
}

void StdXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/stdx/ir/ops.cc.inc" // NOLINT
      >();
}

} // namespace pmlc::dialect::stdx

#define GET_OP_CLASSES
#include "pmlc/dialect/stdx/ir/ops.cc.inc" // NOLINT
