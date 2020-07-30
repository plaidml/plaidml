// Copyright 2019, Intel Corporation

#include "pmlc/dialect/pxa/ir/ops.h"

#include <string>

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

  LogicalResult matchAndRewrite(AffineOpTy affineOp,
                                PatternRewriter &rewriter) const override {
    static_assert(std::is_same<AffineOpTy, AffineReduceOp>::value,
                  "affine reduce op expected");
    auto map = affineOp.getAffineMap();
    AffineMap oldMap = map;
    auto oldOperands = affineOp.getMapOperands();
    SmallVector<Value, 8> resultOperands(oldOperands);
    composeAffineMapAndOperands(&map, &resultOperands);
    if (map == oldMap && std::equal(oldOperands.begin(), oldOperands.end(),
                                    resultOperands.begin()))
      return failure();

    replaceAffineOp(rewriter, affineOp, map, resultOperands);
    return success();
  }
};

template <>
void SimplifyAffineOp<AffineReduceOp>::replaceAffineOp(
    PatternRewriter &rewriter, AffineReduceOp op, AffineMap map,
    ArrayRef<Value> mapOperands) const {
  rewriter.replaceOpWithNewOp<AffineReduceOp>(
      op, op.getMemRefType(), op.agg(), op.val(), op.mem(), map, mapOperands);
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

/// Fold reduce operations with no uses. Reduce has side effects on the heap,
/// but can still be deleted if it has zero uses.
struct SimplifyDeadReduce : public OpRewritePattern<AffineReduceOp> {
  using OpRewritePattern<AffineReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineReduceOp reduce,
                                PatternRewriter &rewriter) const override {
    if (reduce.use_empty()) {
      rewriter.eraseOp(reduce);
      return success();
    }
    return failure();
  }
};

struct SimplifyAffineGemmOp : public OpRewritePattern<AffineGemmOp> {
  using OpRewritePattern<AffineGemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineGemmOp op,
                                PatternRewriter &rewriter) const override {
    auto aAccessMap = op.aAccessMap();
    auto bAccessMap = op.bAccessMap();
    auto cAccessMap = op.cAccessMap();

    SmallVector<Value, 8> aOperands(op.getOperandsForA());
    composeAffineMapAndOperands(&aAccessMap, &aOperands);
    SmallVector<Value, 8> bOperands(op.getOperandsForB());
    composeAffineMapAndOperands(&bAccessMap, &bOperands);
    SmallVector<Value, 8> cOperands(op.getOperandsForC());
    composeAffineMapAndOperands(&cAccessMap, &cOperands);

    SmallVector<Value, 8> mapOperands;
    mapOperands.append(cOperands.begin(), cOperands.end());
    mapOperands.append(aOperands.begin(), aOperands.end());
    mapOperands.append(bOperands.begin(), bOperands.end());

    if (aAccessMap == op.aAccessMap() && bAccessMap == op.bAccessMap() &&
        cAccessMap == op.cAccessMap() &&
        std::equal(mapOperands.begin(), mapOperands.end(),
                   op.mapOperands().begin()))
      return failure();

    rewriter.replaceOpWithNewOp<pxa::AffineGemmOp>(
        op, op.c().getType(),              //
        op.c(), cAccessMap, op.cTileMap(), //
        op.a(), aAccessMap, op.aTileMap(), //
        op.b(), bAccessMap, op.bTileMap(), //
        op.tile(), mapOperands);
    return success();
  }
};

} // namespace

// ---- AffineReduceOp ----

void printAffineReduceOp(OpAsmPrinter &p, AffineReduceOp op) {
  p << op.getOperation()->getName() << ' ';
  p << util::stringifyAggregationKind(op.agg()) << ' ';
  p << op.val() << ", ";
  p << op.mem() << '[';
  auto mapAttr = op.getAttrOfType<AffineMapAttr>("map");
  p.printAffineMapOfSSAIds(mapAttr, op.idxs());
  p << ']';
  p.printOptionalAttrDict(op.getAttrs(), {"agg", "map"});
  p << " : ";
  p.printType(op.mem().getType());
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
      parser.parseColonType(type) || parser.addTypeToList(type, result.types) ||
      parser.resolveOperand(val, type.getElementType(), result.operands) ||
      parser.resolveOperand(out, type, result.operands) ||
      parser.resolveOperands(idxs, indexTy, result.operands));
}

void AffineReduceOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SimplifyAffineGemmOp>(context);
}

OpFoldResult AffineReduceOp::fold(ArrayRef<Attribute> cstOperands) {
  /// reduce(memrefcast) -> reduce
  foldMemRefCast(*this);
  return OpFoldResult();
}

//
// ---- AffineGemmOp ----
//

AffineGemmOp::operand_range AffineGemmOp::getOperandsForA() {
  return getOperands().slice(3 + cAccessMap().getNumInputs(),
                             aAccessMap().getNumInputs());
}

AffineGemmOp::operand_range AffineGemmOp::getOperandsForB() {
  return getOperands().slice(3 + cAccessMap().getNumInputs() +
                                 aAccessMap().getNumInputs(),
                             bAccessMap().getNumInputs());
}

AffineGemmOp::operand_range AffineGemmOp::getOperandsForC() {
  return getOperands().slice(3, cAccessMap().getNumInputs());
}

void AffineGemmOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SimplifyAffineOp<AffineReduceOp>, SimplifyDeadReduce>(context);
}

void printAffineGemmOp(OpAsmPrinter &p, AffineGemmOp op) {
  auto funcType = FunctionType::get({op.a().getType(), op.b().getType()},
                                    {op.c().getType()}, op.getContext());
  p << op.getOperation()->getName() << ' ';
  p << op.c() << '[';
  p.printAffineMapOfSSAIds(op.cAccessMapAttr(), op.getOperandsForC());
  p << "]:";
  p.printAttribute(op.cTileMapAttr());
  p << " = " << op.a() << '[';
  p.printAffineMapOfSSAIds(op.aAccessMapAttr(), op.getOperandsForA());
  p << "]:";
  p.printAttribute(op.aTileMapAttr());
  p << ", " << op.b() << '[';
  p.printAffineMapOfSSAIds(op.bAccessMapAttr(), op.getOperandsForB());
  p << "]:";
  p.printAttribute(op.bTileMapAttr());
  p << ", " << op.tile() << " : " << funcType;
}

struct GemmOperandParser {
  OpAsmParser::OperandType operand;
  SmallVector<OpAsmParser::OperandType, 4> accessOperands;
  AffineMapAttr accessMapAttr;
  AffineMapAttr tileMapAttr;
  std::string accessMapAttrName;
  std::string tileMapAttrName;

  explicit GemmOperandParser(StringRef name)
      : accessMapAttrName(name.str() + "AccessMap"),
        tileMapAttrName(name.str() + "TileMap") {}

  ParseResult parse(OpAsmParser &parser, OperationState &result) {
    return failure(
        parser.parseOperand(operand) ||
        parser.parseAffineMapOfSSAIds(accessOperands, accessMapAttr,
                                      accessMapAttrName, result.attributes) ||
        parser.parseColon() ||
        parser.parseAttribute(tileMapAttr, tileMapAttrName, result.attributes));
  }
};

ParseResult parseAffineGemmOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  auto indexType = builder.getIndexType();
  auto i64Type = builder.getIntegerType(64);
  GemmOperandParser a("a"), b("b"), c("c");
  ArrayAttr tileAttr;
  FunctionType funcType;
  return failure(
      c.parse(parser, result) || parser.parseEqual() ||
      a.parse(parser, result) || parser.parseComma() ||
      b.parse(parser, result) || parser.parseComma() ||
      parser.parseAttribute(tileAttr, i64Type, "tile", result.attributes) ||
      parser.parseColonType(funcType) ||
      parser.addTypesToList(funcType.getResults(), result.types) ||
      parser.resolveOperand(c.operand, funcType.getResult(0),
                            result.operands) ||
      parser.resolveOperand(a.operand, funcType.getInput(0), result.operands) ||
      parser.resolveOperand(b.operand, funcType.getInput(1), result.operands) ||
      parser.resolveOperands(c.accessOperands, indexType, result.operands) ||
      parser.resolveOperands(a.accessOperands, indexType, result.operands) ||
      parser.resolveOperands(b.accessOperands, indexType, result.operands));
}

#define GET_OP_CLASSES
#include "pmlc/dialect/pxa/ir/ops.cc.inc"

PXADialect::PXADialect(mlir::MLIRContext *ctx)
    : mlir::Dialect(getDialectNamespace(), ctx) {
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/pxa/ir/ops.cc.inc" // NOLINT
      >();
}

} // namespace pmlc::dialect::pxa
