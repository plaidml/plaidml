// Copyright 2020, Intel Corporation

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

#include "pmlc/dialect/abi/ir/dialect.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::abi {

void LoopOp::build(::mlir::OpBuilder &odsBuilder,
                   ::mlir::OperationState &odsState) {
  odsState.addRegion();
  odsState.addRegion();
  odsState.addRegion();
}

mlir::Block *LoopOp::getBodyEntryBlock() { return &bodyRegion().front(); }
mlir::Block *LoopOp::getFiniEntryBlock() { return &finiRegion().front(); }

YieldOp LoopOp::getInitTerminator() {
  return mlir::cast<YieldOp>(initRegion().back().getTerminator());
}

TerminatorOp LoopOp::getFiniTerminator() {
  return mlir::cast<TerminatorOp>(finiRegion().back().getTerminator());
}

static void printFuncLikeRegion(mlir::OpAsmPrinter &p, Region &body) {
  p << "(";
  for (unsigned i = 0; i < body.getNumArguments(); i++) {
    if (i != 0)
      p << ", ";
    p.printOperand(body.getArgument(i));
    p << ": ";
    p.printType(body.getArgument(i).getType());
  }
  p << ")";
  p.printRegion(body, /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);
}

// TODO: This code somewhat duplicates mlir::impl::parseFunctionArgumentList,
// which is public on later versions of upstream (but private as of this commit)
static ParseResult
parseFunctionArgumentList(OpAsmParser &parser,
                          SmallVectorImpl<OpAsmParser::OperandType> &argNames,
                          SmallVectorImpl<Type> &argTypes) {
  if (parser.parseLParen())
    return failure();

  if (failed(parser.parseOptionalRParen())) {
    do {
      OpAsmParser::OperandType argument;
      Type argumentType;
      if (failed(parser.parseRegionArgument(argument)))
        return failure();
      if (failed(parser.parseColonType(argumentType)))
        return failure();
      argNames.push_back(argument);
      argTypes.push_back(argumentType);
    } while (succeeded(parser.parseOptionalComma()));
    parser.parseRParen();
  }

  return success();
}

static ParseResult parseFuncLikeRegion(OpAsmParser &parser, Region &body) {
  SmallVector<OpAsmParser::OperandType, 4> argNames;
  SmallVector<Type, 4> argTypes;
  if (failed(parseFunctionArgumentList(parser, argNames, argTypes)))
    return failure();
  return parser.parseRegion(body, argNames, argTypes);
}

static void printLoopOp(mlir::OpAsmPrinter &p, LoopOp op) {
  p << "abi.loop init";
  printFuncLikeRegion(p, op.initRegion());
  p << " body";
  printFuncLikeRegion(p, op.bodyRegion());
  p << " fini";
  printFuncLikeRegion(p, op.finiRegion());
  p.printOptionalAttrDict(op.getAttrs(), /*elidedAttrs=*/{});
}

static ParseResult parseLoopOp(OpAsmParser &parser, OperationState &result) {
  if (failed(parser.parseKeyword("init")))
    return failure();
  Region *initRegion = result.addRegion();
  if (failed(parseFuncLikeRegion(parser, *initRegion)))
    return failure();
  if (failed(parser.parseKeyword("body")))
    return failure();
  Region *bodyRegion = result.addRegion();
  if (failed(parseFuncLikeRegion(parser, *bodyRegion)))
    return failure();
  if (failed(parser.parseKeyword("fini")))
    return failure();
  Region *finiRegion = result.addRegion();
  if (failed(parseFuncLikeRegion(parser, *finiRegion)))
    return failure();
  return parser.parseOptionalAttrDict(result.attributes);
}

} // namespace pmlc::dialect::abi

#define GET_OP_CLASSES
#include "pmlc/dialect/abi/ir/ops.cc.inc"
