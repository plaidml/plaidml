// Copyright 2020 Intel Corporation

#include "pmlc/dialect/stdx/ir/ops.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::stdx {

// ---- ReshapeOp ----

Value ReshapeOp::getViewSource() { return tensor(); }

LogicalResult ReshapeOp::verify() { return success(); }

// ---- SubgroupBlockReadINTELOp ----

void SubgroupBlockReadINTELOp::build(OpBuilder &builder, OperationState &result,
                                     Value memref, ValueRange indices) {
  auto memrefType = memref.getType().cast<MemRefType>();
  result.addOperands(memref);
  result.addOperands(indices);
  result.addTypes(memrefType.getElementType());
}

LogicalResult
SubgroupBlockReadINTELOp::verify() {
  if (getNumOperands() != 1 + getMemRefType().getRank())
    return emitOpError("incorrect number of indices for memory ref");
  return success();
}

// ---- SubgroupBlockWriteINTELOp ----

void SubgroupBlockWriteINTELOp::build(OpBuilder &builder,
                                      OperationState &result,
                                      Value valueToStore, Value memref) {
  result.addOperands(valueToStore);
  result.addOperands(memref);
}

LogicalResult
SubgroupBlockWriteINTELOp::verify() {
  if (getNumOperands() != 2 + getMemRefType().getRank())
    return emitOpError("subgroup block write index operand"
                          " count not equal to memref rank");

  return success();
}

// ---- ClosureOp ----

ParseResult ClosureOp::parse(OpAsmParser &parser, OperationState &result) {
  MLIRContext *ctx = result.getContext();
  Builder &builder = parser.getBuilder();

  SmallVector<OpAsmParser::Argument> entryArgs;
  SmallVector<DictionaryAttr> resultAttrs;
  SmallVector<Type> resultTypes;
  bool isVariadic;

  if (failed(function_interface_impl::parseFunctionSignature(
          parser, /*allowVariadic=*/false, entryArgs, isVariadic, resultTypes, resultAttrs)))
    return failure();

  // Parse operation attributes.
  NamedAttrList attrs;
  if (parser.parseOptionalAttrDictWithKeyword(attrs))
    return failure();

  SmallVector<Type> argTypes;
  for (auto &arg : entryArgs)
    argTypes.push_back(arg.type);
  
  result.addAttributes(attrs);
  result.addAttribute(
      "type", TypeAttr::get(builder.getFunctionType(argTypes, resultTypes)));

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/entryArgs, 
                         /*enableNameShadowing=*/false))
    return failure();
  return success();
}

void ClosureOp::print(OpAsmPrinter &p) {
  FunctionType type = getFunctionType();
  function_interface_impl::printFunctionSignature(p, *this, type.getInputs(),
                                             /*isVariadic=*/false,
                                             type.getResults());
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), {"type"});
  p.printRegion(body(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

Region &ClosureOp::getLoopBody() { return body(); }

LogicalResult ClosureOp::verify() {
  // TODO
  return success();
}

// ---- YieldOp ----

LogicalResult YieldOp::verify() {
  // TODO
  return success();
}

// ---- SubgroupBroadcastOp ----

LogicalResult SubgroupBroadcastOp::verify() {
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
