// Copyright 2019, Intel Corporation

#include "pmlc/dialect/pxa/ir/ops.h"

#include "mlir/IR/OpImplementation.h"

namespace pmlc::dialect::pxa {

using llvm::SmallVector;
using mlir::AffineExpr;
using mlir::AffineMap;
using mlir::AffineMapAttr;
using mlir::Block;
using mlir::failure;
using mlir::OpAsmParser;
using mlir::OpAsmPrinter;
using mlir::success;

namespace {

template <typename Symbolizer>
ParseResult parseKeywordIntoEnumAttr(OpAsmParser& parser, OperationState& result, StringRef attrName, Type attrType,
                                     Symbolizer symbolizer) {
  llvm::SMLoc loc;
  StringRef keyword;
  if (parser.getCurrentLocation(&loc) || parser.parseKeyword(&keyword)) {
    return failure();
  }

  auto enumValue = symbolizer(keyword);
  if (!enumValue) {
    return parser.emitError(loc) << "'" << keyword << "' is an incorrect value of the '" << attrName << "' attribute";
  }

  auto intValue = static_cast<int64_t>(enumValue.getValue());
  auto attr = parser.getBuilder().getIntegerAttr(attrType, intValue);
  result.addAttribute(attrName, attr);

  return success();
}

}  // namespace

void AffineParallelForOp::build(Builder* builder, OperationState& result, ArrayRef<int64_t> ranges) {
  SmallVector<AffineExpr, 8> range_exprs;
  // Make range expressions for each range
  for (int64_t range : ranges) {
    range_exprs.push_back(builder->getAffineConstantExpr(range));
  }
  // Make the maps (handling the 0-dim case carefully)
  if (ranges.size()) {
    result.addAttribute("ranges", AffineMapAttr::get(AffineMap::get(0, 0, range_exprs)));
    result.addAttribute("transform", AffineMapAttr::get(builder->getMultiDimIdentityMap(ranges.size())));
  } else {
    result.addAttribute("ranges", AffineMapAttr::get(AffineMap::get(builder->getContext())));
    result.addAttribute("transform", AffineMapAttr::get(AffineMap::get(builder->getContext())));
  }
  // Create a region and a block for the body.
  auto bodyRegion = result.addRegion();
  auto body = new Block();
  // Add all the args
  for (size_t i = 0; i < ranges.size(); i++) {
    body->addArgument(IndexType::get(builder->getContext()));
  }
  bodyRegion->push_back(body);
  // Terminate
  ensureTerminator(*bodyRegion, *builder, result.location);
}

// ---- AffineReduceOp ----

void printAffineReduceOp(OpAsmPrinter& p, AffineReduceOp op) {
  p << op.getOperation()->getName() << ' ';
  p << util::stringifyAggregationKind(op.agg()) << ' ';
  p << op.val() << ", ";
  p << op.out() << '[';
  auto mapAttr = op.getAttrOfType<AffineMapAttr>("map");
  p.printAffineMapOfSSAIds(mapAttr, op.idxs());
  p << ']';
  p.printOptionalAttrDict(op.getAttrs(), {"agg", "map"});
  p << " : ";
  p.printType(op.out().getType());
}

// <operation> ::= `pxa.reduce` keyword ssa-use `,` ssa-use `[` ssa-use-list `]`
//                 attribute-dict? `:` type
ParseResult parseAffineReduceOp(OpAsmParser& parser, OperationState& result) {
  auto indexTy = parser.getBuilder().getIndexType();
  auto i64Ty = parser.getBuilder().getIntegerType(64);
  MemRefType type;
  AffineMapAttr mapAttr;
  OpAsmParser::OperandType val, out;
  SmallVector<OpAsmParser::OperandType, 4> idxs;
  auto symbolizeAggregationKind = [](StringRef str) { return util::symbolizeAggregationKind(str); };
  return failure(parseKeywordIntoEnumAttr(parser, result, "agg", i64Ty, symbolizeAggregationKind) ||
                 parser.parseOperand(val) || parser.parseComma() || parser.parseOperand(out) ||
                 parser.parseAffineMapOfSSAIds(idxs, mapAttr, "map", result.attributes) ||
                 parser.parseOptionalAttrDict(result.attributes) || parser.parseColonType(type) ||
                 parser.resolveOperand(val, type.getElementType(), result.operands) ||
                 parser.resolveOperand(out, type, result.operands) ||
                 parser.resolveOperands(idxs, indexTy, result.operands));
}

#define GET_OP_CLASSES
#include "pmlc/dialect/pxa/ir/ops.cc.inc"

}  // namespace pmlc::dialect::pxa
