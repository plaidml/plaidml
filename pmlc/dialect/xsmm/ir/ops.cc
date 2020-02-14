// Copyright 2020 Intel Corporation

#include "pmlc/dialect/xsmm/ir/ops.h"

#include "mlir/IR/OpImplementation.h"

namespace pmlc::dialect::xsmm {

using llvm::SmallVector;
using mlir::failure;
using mlir::success;

namespace {

// ---- SMMDispatchOp ----

void printSMMDispatchOp(OpAsmPrinter &p, SMMDispatchOp op) {
  // p << op.getOperation()->getName() << ' ' << op.getInductionVar();
  // p << " = " << op.memref() << '[' << op.indices() << ']';
  // p.printOptionalAttrDict(op.getAttrs());
  // p << " : " << op.memref().getType();
  // p.printRegion(op.body(), /*printEntryBlockArgs=*/false);
}

// <operation> ::= `xsmm.atomic_rmw` argument `=` ssa-use `[` ssa-use-list `]`
//                 attribute-dict? `:` type region
ParseResult parseSMMDispatchOp(OpAsmParser &parser, OperationState &result) {
  // MemRefType type;
  // OpAsmParser::OperandType iv;
  // OpAsmParser::OperandType memref;
  // auto indexTy = parser.getBuilder().getIndexType();
  // SmallVector<OpAsmParser::OperandType, 4> idxs;
  // Region *body = result.addRegion();
  // if (parser.parseRegionArgument(iv) || parser.parseEqual() ||
  //     parser.parseOperand(memref) ||
  //     parser.parseOperandList(idxs, OpAsmParser::Delimiter::Square) ||
  //     parser.parseOptionalAttrDict(result.attributes) ||
  //     parser.parseColonType(type) ||
  //     parser.resolveOperand(memref, type, result.operands) ||
  //     parser.resolveOperands(idxs, indexTy, result.operands) ||
  //     parser.parseRegion(*body, iv, type.getElementType())) {
  //   return failure();
  // }
  // result.addTypes(type.getElementType());
  return success();
}

LogicalResult verifySMMDispatchOp(SMMDispatchOp op) {
  // if (op.getMemRefType().getRank() != op.getNumOperands() - 1)
  //   return op.emitOpError(
  //       "expects the number of subscripts to be equal to memref rank");
  // auto block = op.getBody();
  // if (block->empty()) {
  //   return op.emitOpError("expects a non-empty body");
  // }
  // auto elementType = op.getMemRefType().getElementType();
  // if (block->getNumArguments() != 1 ||
  //     block->getArgument(0).getType() != elementType) {
  //   return op.emitOpError()
  //          << "expects a body with one argument of type " << elementType;
  // }
  // if (!llvm::isa<AtomicRMWYieldOp>(block->getTerminator())) {
  //   return op.emitOpError(
  //       "expects the body to be terminated with a 'xsmm.atomic_rmw.yield'
  //       op");
  // }
  return success();
}

} // namespace

#define GET_OP_CLASSES
#include "pmlc/dialect/xsmm/ir/ops.cc.inc"

} // namespace pmlc::dialect::xsmm
