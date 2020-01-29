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

// ---- AffineParallelOp ----

void AffineParallelOp::build(Builder* builder, OperationState& result, ArrayRef<int64_t> ranges) {
  // Default initalize empty maps
  auto lb_map = AffineMap::get(builder->getContext());
  auto ub_map = AffineMap::get(builder->getContext());
  // If ranges, set to [0, N) for each range
  if (ranges.size()) {
    SmallVector<AffineExpr, 8> lbExprs;
    SmallVector<AffineExpr, 8> ubExprs;
    // Make range expressions for each range
    for (int64_t range : ranges) {
      lbExprs.push_back(builder->getAffineConstantExpr(0));
      ubExprs.push_back(builder->getAffineConstantExpr(range));
    }
    lb_map = AffineMap::get(0, 0, lbExprs);
    ub_map = AffineMap::get(0, 0, ubExprs);
  }
  // Fall through
  build(builder, result, lb_map, {}, ub_map, {});
}

void AffineParallelOp::build(Builder* builder, OperationState& result, AffineMap lb_map, ValueRange lb_args,
                             AffineMap ub_map, ValueRange ub_args) {
  // Verify sizes
  size_t idxCount = lb_map.getNumResults();
  assert(idxCount == ub_map.getNumResults());
  // Make default step sizes of 1
  SmallVector<int64_t, 8> steps(idxCount, 1);
  // Call through
  build(builder, result, lb_map, lb_args, ub_map, ub_args, steps);
}

void AffineParallelOp::build(Builder* builder, OperationState& result, AffineMap lb_map, ValueRange lb_args,
                             AffineMap ub_map, ValueRange ub_args, ArrayRef<int64_t> steps) {
  // Verify sizes
  size_t idxCount = lb_map.getNumResults();
  assert(idxCount == ub_map.getNumResults());
  assert(idxCount == steps.size());
  // Set all of the attributes
  result.addAttribute("lowerBoundsMap", AffineMapAttr::get(lb_map));
  result.addAttribute("upperBoundsMap", AffineMapAttr::get(ub_map));
  result.addAttribute("steps", builder->getI64ArrayAttr(steps));
  result.addOperands(lb_args);
  result.addOperands(ub_args);
  // Create a region and a block for the body.
  auto bodyRegion = result.addRegion();
  auto body = new Block();
  // Add all the args
  for (size_t i = 0; i < idxCount; i++) {
    body->addArgument(IndexType::get(builder->getContext()));
  }
  bodyRegion->push_back(body);
  // Terminate
  ensureTerminator(*bodyRegion, *builder, result.location);
}

AffineParallelOp::operand_range AffineParallelOp::getLowerBoundsOperands() {
  return {operand_begin(), operand_begin() + lowerBoundsMap().getNumInputs()};
}

AffineParallelOp::operand_range AffineParallelOp::getUpperBoundsOperands() {
  return {operand_begin() + lowerBoundsMap().getNumInputs(), operand_end()};
}

mlir::Block* AffineParallelOp::getBody() { return &region().front(); }

mlir::OpBuilder AffineParallelOp::getBodyBuilder() { return mlir::OpBuilder(getBody(), std::prev(getBody()->end())); }

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
