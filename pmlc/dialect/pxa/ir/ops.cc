// Copyright 2019, Intel Corporation

#include "pmlc/dialect/pxa/ir/ops.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

namespace pmlc::dialect::pxa {

using namespace mlir; // NOLINT

namespace {

template <typename Symbolizer>
ParseResult parseKeywordIntoEnumAttr(OpAsmParser &parser,
                                     OperationState &result, StringRef attrName,
                                     Type attrType, Symbolizer symbolizer) {
  llvm::SMLoc loc;
  StringRef keyword;
  if (parser.getCurrentLocation(&loc) || parser.parseKeyword(&keyword)) {
    return failure();
  }

  auto enumValue = symbolizer(keyword);
  if (!enumValue) {
    return parser.emitError(loc)
           << "'" << keyword << "' is an incorrect value of the '" << attrName
           << "' attribute";
  }

  auto intValue = static_cast<int64_t>(enumValue.getValue());
  auto attr = parser.getBuilder().getIntegerAttr(attrType, intValue);
  result.addAttribute(attrName, attr);

  return success();
}

/// Implements `map` and `operands` composition and simplification to support
/// `makeComposedAffineApply`. This can be called to achieve the same effects
/// on `map` and `operands` without creating an AffineApplyOp that needs to be
/// immediately deleted.
static void composeAffineMapAndOperands(AffineMap *map,
                                        SmallVectorImpl<Value> *operands) {
  AffineApplyNormalizer normalizer(*map, *operands);
  auto normalizedMap = normalizer.getAffineMap();
  auto normalizedOperands = normalizer.getOperands();
  canonicalizeMapAndOperands(&normalizedMap, &normalizedOperands);
  *map = normalizedMap;
  *operands = normalizedOperands;
  assert(*map);
}

/// Simplify AffineApply, AffineLoad, and AffineStore operations by composing
/// maps that supply results into them.
///
template <typename AffineOpTy>
struct SimplifyAffineOp : public OpRewritePattern<AffineOpTy> {
  using OpRewritePattern<AffineOpTy>::OpRewritePattern;

  /// Replace the affine op with another instance of it with the supplied
  /// map and mapOperands.
  void replaceAffineOp(PatternRewriter &rewriter, AffineOpTy affineOp,
                       AffineMap map, ArrayRef<Value> mapOperands) const;

  PatternMatchResult matchAndRewrite(AffineOpTy affineOp,
                                     PatternRewriter &rewriter) const override {
    static_assert(
        // std::is_same<AffineOpTy, AffineLoadOp>::value ||
        // std::is_same<AffineOpTy, AffinePrefetchOp>::value ||
        // std::is_same<AffineOpTy, AffineStoreOp>::value ||
        std::is_same<AffineOpTy, AffineApplyOp>::value ||
            std::is_same<AffineOpTy, AffineReduceOp>::value,
        "affine load/store/apply/reduce op expected");
    auto map = affineOp.getAffineMap();
    AffineMap oldMap = map;
    auto oldOperands = affineOp.getMapOperands();
    SmallVector<Value, 8> resultOperands(oldOperands);
    composeAffineMapAndOperands(&map, &resultOperands);
    if (map == oldMap && std::equal(oldOperands.begin(), oldOperands.end(),
                                    resultOperands.begin()))
      return this->matchFailure();

    replaceAffineOp(rewriter, affineOp, map, resultOperands);
    return this->matchSuccess();
  }
};

// Specialize the template to account for the different build signatures for
// affine load, store, reduce, and apply ops.
template <>
void SimplifyAffineOp<AffineReduceOp>::replaceAffineOp(
    PatternRewriter &rewriter, AffineReduceOp op, AffineMap map,
    ArrayRef<Value> mapOperands) const {
  rewriter.replaceOpWithNewOp<AffineReduceOp>(op, op.agg(), op.val(), op.out(),
                                              map, mapOperands);
}

/// This is a common class used for patterns of the form
/// "someop(memrefcast) -> someop".  It folds the source of any memref_cast
/// into the root operation directly.
static LogicalResult foldMemRefCast(Operation *op) {
  bool folded = false;
  for (OpOperand &operand : op->getOpOperands()) {
    auto cast = dyn_cast_or_null<MemRefCastOp>(operand.get().getDefiningOp());
    if (cast && !cast.getOperand().getType().isa<UnrankedMemRefType>()) {
      operand.set(cast.getOperand());
      folded = true;
    }
  }
  return success(folded);
}

} // namespace

// ---- AffineReduceOp ----

void printAffineReduceOp(OpAsmPrinter &p, AffineReduceOp op) {
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
ParseResult parseAffineReduceOp(OpAsmParser &parser, OperationState &result) {
  auto indexTy = parser.getBuilder().getIndexType();
  auto i64Ty = parser.getBuilder().getIntegerType(64);
  MemRefType type;
  AffineMapAttr mapAttr;
  OpAsmParser::OperandType val, out;
  SmallVector<OpAsmParser::OperandType, 4> idxs;
  auto symbolizeAggregationKind = [](StringRef str) {
    return util::symbolizeAggregationKind(str);
  };
  return failure(
      parseKeywordIntoEnumAttr(parser, result, "agg", i64Ty,
                               symbolizeAggregationKind) ||
      parser.parseOperand(val) || parser.parseComma() ||
      parser.parseOperand(out) ||
      parser.parseAffineMapOfSSAIds(idxs, mapAttr, "map", result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(val, type.getElementType(), result.operands) ||
      parser.resolveOperand(out, type, result.operands) ||
      parser.resolveOperands(idxs, indexTy, result.operands));
}

void AffineReduceOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SimplifyAffineOp<AffineReduceOp>>(context);
}

LogicalResult AffineReduceOp::fold(ArrayRef<Attribute> cstOperands,
                                   SmallVectorImpl<OpFoldResult> &results) {
  /// reduce(memrefcast) -> reduce
  return foldMemRefCast(*this);
}

#define GET_OP_CLASSES
#include "pmlc/dialect/pxa/ir/ops.cc.inc"

} // namespace pmlc::dialect::pxa
