// Copyright 2020 Intel Corporation

#include "pmlc/dialect/stdx/ir/ops.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
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
  result.addTypes(memrefType.getElementType());
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

// ---- ClosureOp ----

static ParseResult parseClosureOp(OpAsmParser &parser, OperationState &result) {
  MLIRContext *ctx = result.getContext();
  Builder &builder = parser.getBuilder();

  SmallVector<OpAsmParser::OperandType, 4> args;
  SmallVector<NamedAttrList, 4> argAttrs;
  SmallVector<NamedAttrList, 4> resultAttrs;
  SmallVector<Type, 4> argTypes;
  SmallVector<Type, 4> resultTypes;

  bool isVariadic = false;
  if (function_like_impl::parseFunctionSignature(
          parser, /*allowVariadic=*/false, args, argTypes, argAttrs, isVariadic,
          resultTypes, resultAttrs))
    return failure();

  // Parse operation attributes.
  NamedAttrList attrs;
  if (parser.parseOptionalAttrDictWithKeyword(attrs))
    return failure();

  result.addAttributes(attrs);
  result.addAttribute(
      "type", TypeAttr::get(builder.getFunctionType(argTypes, resultTypes)));

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/{args},
                         /*argTypes=*/{argTypes},
                         /*enableNameShadowing=*/false))
    return failure();
  return success();
}

static void printClosureOp(OpAsmPrinter &p, ClosureOp op) {
  FunctionType type = op.getType();
  p << op.getOperationName();
  function_like_impl::printFunctionSignature(p, op, type.getInputs(),
                                             /*isVariadic=*/false,
                                             type.getResults());
  p.printOptionalAttrDictWithKeyword(op->getAttrs(), {"type"});
  p.printRegion(op.body(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

Region &ClosureOp::getLoopBody() { return body(); }

bool ClosureOp::isDefinedOutsideOfLoop(Value value) {
  return !body().isAncestor(value.getParentRegion());
}

LogicalResult ClosureOp::moveOutOfLoop(ArrayRef<Operation *> ops) {
  for (Operation *op : ops)
    op->moveBefore(*this);
  return success();
}

static LogicalResult verifyClosureOp(ClosureOp op) {
  // TODO
  return success();
}

// ---- YieldOp ----

static LogicalResult verifyYieldOp(YieldOp op) {
  // TODO
  return success();
}

// ---- StdXDialect ----

void StdXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/stdx/ir/ops.cc.inc" // NOLINT
      >();
}

} // namespace pmlc::dialect::stdx

#include "pmlc/dialect/stdx/ir/dialect.cc.inc" // NOLINT

#define GET_OP_CLASSES
#include "pmlc/dialect/stdx/ir/ops.cc.inc" // NOLINT
