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

/// Simplify operations by composing maps that supply results into them.
template <typename AffineOpTy>
struct SimplifyAffineOp : public OpRewritePattern<AffineOpTy> {
  using OpRewritePattern<AffineOpTy>::OpRewritePattern;

  /// Replace the affine op with another instance of it with the supplied
  /// map and mapOperands.
  void replaceAffineOp(PatternRewriter &rewriter, AffineOpTy affineOp,
                       AffineMap map, ArrayRef<Value> mapOperands) const;

  LogicalResult matchAndRewrite(AffineOpTy affineOp,
                                PatternRewriter &rewriter) const override {
    static_assert(std::is_same<AffineOpTy, PxaReduceOp>::value ||
                      std::is_same<AffineOpTy, PxaVectorReduceOp>::value ||
                      std::is_same<AffineOpTy, PxaLoadOp>::value ||
                      std::is_same<AffineOpTy, PxaVectorLoadOp>::value,
                  "affine reduce/vector_reduce or load op expected");
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
void SimplifyAffineOp<PxaLoadOp>::replaceAffineOp(
    PatternRewriter &rewriter, PxaLoadOp load, AffineMap map,
    ArrayRef<Value> mapOperands) const {
  rewriter.replaceOpWithNewOp<PxaLoadOp>(load, load.getMemRef(), map,
                                         mapOperands);
}

template <>
void SimplifyAffineOp<PxaReduceOp>::replaceAffineOp(
    PatternRewriter &rewriter, PxaReduceOp op, AffineMap map,
    ArrayRef<Value> mapOperands) const {
  rewriter.replaceOpWithNewOp<PxaReduceOp>(op, op.getMemRefType(), op.agg(),
                                           op.val(), op.memref(), map,
                                           mapOperands);
}

template <>
void SimplifyAffineOp<PxaVectorLoadOp>::replaceAffineOp(
    PatternRewriter &rewriter, PxaVectorLoadOp op, AffineMap map,
    ArrayRef<Value> mapOperands) const {
  rewriter.replaceOpWithNewOp<PxaVectorLoadOp>(
      op, op.getVectorType(), op.getMemRef(), map, mapOperands);
}
template <>
void SimplifyAffineOp<PxaVectorReduceOp>::replaceAffineOp(
    PatternRewriter &rewriter, PxaVectorReduceOp op, AffineMap map,
    ArrayRef<Value> mapOperands) const {
  rewriter.replaceOpWithNewOp<PxaVectorReduceOp>(op, op.getMemRefType(),
                                                 op.agg(), op.vector(),
                                                 op.memref(), map, mapOperands);
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
template <typename ReduceOp>
struct SimplifyDeadReduce : public OpRewritePattern<ReduceOp> {
  using OpRewritePattern<ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp reduce,
                                PatternRewriter &rewriter) const override {
    if (reduce.use_empty()) {
      rewriter.eraseOp(reduce);
      return success();
    }
    return failure();
  }
};

struct SimplifyPxaGemmOp : public OpRewritePattern<PxaGemmOp> {
  using OpRewritePattern<PxaGemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PxaGemmOp op,
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

    rewriter.replaceOpWithNewOp<pxa::PxaGemmOp>(
        op, op.c().getType(),              //
        op.c(), cAccessMap, op.cTileMap(), //
        op.a(), aAccessMap, op.aTileMap(), //
        op.b(), bAccessMap, op.bTileMap(), //
        op.tile(), op.numBatches(), mapOperands);
    return success();
  }
};
} // namespace

// ---- PxaLoadOp ----

void PxaLoadOp::build(OpBuilder &builder, OperationState &result, AffineMap map,
                      ValueRange operands) {
  assert(operands.size() == 1 + map.getNumInputs() && "inconsistent operands");
  result.addOperands(operands);
  if (map)
    result.addAttribute(getMapAttrName(), AffineMapAttr::get(map));
  auto memrefType = operands[0].getType().cast<MemRefType>();
  result.types.push_back(memrefType.getElementType());
}

void PxaLoadOp::build(OpBuilder &builder, OperationState &result, Value memref,
                      AffineMap map, ValueRange mapOperands) {
  assert(map.getNumInputs() == mapOperands.size() && "inconsistent index info");
  result.addOperands(memref);
  result.addOperands(mapOperands);
  auto memrefType = memref.getType().cast<MemRefType>();
  result.addAttribute(getMapAttrName(), AffineMapAttr::get(map));
  result.types.push_back(memrefType.getElementType());
}

void PxaLoadOp::build(OpBuilder &builder, OperationState &result, Value memref,
                      ValueRange indices) {
  auto memrefType = memref.getType().cast<MemRefType>();
  auto rank = memrefType.getRank();
  // Create identity map for memrefs with at least one dimension or () -> ()
  // for zero-dimensional memrefs.
  auto map =
      rank ? builder.getMultiDimIdentityMap(rank) : builder.getEmptyAffineMap();
  build(builder, result, memref, map, indices);
}

static void printPxaLoadOp(OpAsmPrinter &p, PxaLoadOp op) {
  p << "pxa.load " << op.getMemRef() << '[';
  if (AffineMapAttr mapAttr =
          op->getAttrOfType<AffineMapAttr>(op.getMapAttrName()))
    p.printAffineMapOfSSAIds(mapAttr, op.getMapOperands());
  p << ']';
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{op.getMapAttrName()});
  p << " : " << op.getMemRefType();
}

static ParseResult parsePxaLoadOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  auto indexTy = builder.getIndexType();

  MemRefType type;
  OpAsmParser::OperandType memrefInfo;
  AffineMapAttr mapAttr;
  SmallVector<OpAsmParser::OperandType, 1> mapOperands;
  return failure(
      parser.parseOperand(memrefInfo) ||
      parser.parseAffineMapOfSSAIds(mapOperands, mapAttr,
                                    PxaLoadOp::getMapAttrName(),
                                    result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type) ||
      parser.resolveOperand(memrefInfo, type, result.operands) ||
      parser.resolveOperands(mapOperands, indexTy, result.operands) ||
      parser.addTypeToList(type.getElementType(), result.types));
}

void PxaLoadOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  results.insert<SimplifyAffineOp<PxaLoadOp>>(context);
}

OpFoldResult PxaLoadOp::fold(ArrayRef<Attribute> cstOperands) {
  /// load(memrefcast) -> load
  if (succeeded(foldMemRefCast(*this)))
    return getResult();
  return OpFoldResult();
}

// ---- PxaVectorLoadOp ----

void PxaVectorLoadOp::build(OpBuilder &builder, OperationState &result,
                            VectorType type, Value memref, AffineMap map,
                            ValueRange mapOperands) {
  assert(map.getNumInputs() == mapOperands.size() && "inconsistent index info");
  result.addOperands(memref);
  result.addOperands(mapOperands);
  result.addAttribute(getMapAttrName(), AffineMapAttr::get(map));
  result.types.push_back(type);
}

static void printPxaVectorLoadOp(OpAsmPrinter &p, PxaVectorLoadOp op) {
  p << "pxa.vector_load " << op.getMemRef() << '[';
  if (AffineMapAttr mapAttr =
          op->getAttrOfType<AffineMapAttr>(op.getMapAttrName()))
    p.printAffineMapOfSSAIds(mapAttr, op.getMapOperands());
  p << ']';
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{op.getMapAttrName()});
  p << " : " << op.getMemRefType() << ", " << op.getType();
}

static ParseResult parsePxaVectorLoadOp(OpAsmParser &parser,
                                        OperationState &result) {
  auto &builder = parser.getBuilder();
  auto indexTy = builder.getIndexType();

  MemRefType memrefType;
  VectorType resultType;
  OpAsmParser::OperandType memrefInfo;
  AffineMapAttr mapAttr;
  SmallVector<OpAsmParser::OperandType, 1> mapOperands;
  return failure(
      parser.parseOperand(memrefInfo) ||
      parser.parseAffineMapOfSSAIds(mapOperands, mapAttr,
                                    PxaVectorLoadOp::getMapAttrName(),
                                    result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(memrefType) || parser.parseComma() ||
      parser.parseType(resultType) ||
      parser.resolveOperand(memrefInfo, memrefType, result.operands) ||
      parser.resolveOperands(mapOperands, indexTy, result.operands) ||
      parser.addTypeToList(resultType, result.types));
}

void PxaVectorLoadOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SimplifyAffineOp<PxaVectorLoadOp>>(context);
}

OpFoldResult PxaVectorLoadOp::fold(ArrayRef<Attribute> cstOperands) {
  /// reduce(memrefcast) -> reduce
  foldMemRefCast(*this);
  return OpFoldResult();
}

// ---- PxaReduceOp ----

void printPxaReduceOp(OpAsmPrinter &p, PxaReduceOp op) {
  p << op.getOperation()->getName() << ' ';
  p << stringifyAtomicRMWKind(op.agg()) << ' ';
  p << op.val() << ", ";
  p << op.memref() << '[';
  auto mapAttr = op->getAttrOfType<AffineMapAttr>("map");
  p.printAffineMapOfSSAIds(mapAttr, op.idxs());
  p << ']';
  p.printOptionalAttrDict(op->getAttrs(), {"agg", "map"});
  p << " : ";
  p.printType(op.memref().getType());
}

// <operation> ::= `pxa.reduce` keyword ssa-use `,` ssa-use `[` ssa-use-list `]`
//                 attribute-dict? `:` type
ParseResult parsePxaReduceOp(OpAsmParser &parser, OperationState &result) {
  auto indexTy = parser.getBuilder().getIndexType();
  auto i64Ty = parser.getBuilder().getIntegerType(64);
  MemRefType type;
  AffineMapAttr mapAttr;
  OpAsmParser::OperandType val, out;
  SmallVector<OpAsmParser::OperandType, 4> idxs;
  auto symbolizeAtomicRMWKindWrap = [](StringRef str) {
    return symbolizeAtomicRMWKind(str);
  };
  return failure(
      parseKeywordIntoEnumAttr(parser, result, "agg", i64Ty,
                               symbolizeAtomicRMWKindWrap) ||
      parser.parseOperand(val) || parser.parseComma() ||
      parser.parseOperand(out) ||
      parser.parseAffineMapOfSSAIds(idxs, mapAttr, "map", result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type) || parser.addTypeToList(type, result.types) ||
      parser.resolveOperand(val, type.getElementType(), result.operands) ||
      parser.resolveOperand(out, type, result.operands) ||
      parser.resolveOperands(idxs, indexTy, result.operands));
}

void PxaReduceOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                              MLIRContext *context) {
  results.insert<                    //
      SimplifyAffineOp<PxaReduceOp>, //
      SimplifyDeadReduce<PxaReduceOp>>(context);
}

OpFoldResult PxaReduceOp::fold(ArrayRef<Attribute> cstOperands) {
  /// reduce(memrefcast) -> reduce
  foldMemRefCast(*this);
  return OpFoldResult();
}

//
// ---- PxaGemmOp ----
//

PxaGemmOp::operand_range PxaGemmOp::getOperandsForA() {
  return getOperands().slice(3 + cAccessMap().getNumInputs(),
                             aAccessMap().getNumInputs());
}

PxaGemmOp::operand_range PxaGemmOp::getOperandsForB() {
  return getOperands().slice(3 + cAccessMap().getNumInputs() +
                                 aAccessMap().getNumInputs(),
                             bAccessMap().getNumInputs());
}

PxaGemmOp::operand_range PxaGemmOp::getOperandsForC() {
  return getOperands().slice(3, cAccessMap().getNumInputs());
}

void PxaGemmOp::getCanonicalizationPatterns(OwningRewritePatternList &results,
                                            MLIRContext *context) {
  results.insert<SimplifyPxaGemmOp>(context);
}

void printPxaGemmOp(OpAsmPrinter &p, PxaGemmOp op) {
  auto funcType =
      FunctionType::get(op.getContext(), {op.a().getType(), op.b().getType()},
                        {op.c().getType()});
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
  p << ", " << op.tile() << ", " << op.numBatches() << " : " << funcType;
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

ParseResult parsePxaGemmOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  auto indexType = builder.getIndexType();
  auto i64Type = builder.getIntegerType(64);
  GemmOperandParser a("a"), b("b"), c("c");
  ArrayAttr tileAttr;
  IntegerAttr numBatchesAttr;
  FunctionType funcType;
  return failure(
      c.parse(parser, result) || parser.parseEqual() ||
      a.parse(parser, result) || parser.parseComma() ||
      b.parse(parser, result) || parser.parseComma() ||
      parser.parseAttribute(tileAttr, i64Type, "tile", result.attributes) ||
      parser.parseComma() ||
      parser.parseAttribute(numBatchesAttr, i64Type, "numBatches",
                            result.attributes) ||
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

// ---- PxaVectorReduceOp ----

void printPxaVectorReduceOp(OpAsmPrinter &p, PxaVectorReduceOp op) {
  p << op.getOperation()->getName() << ' ';
  p << stringifyAtomicRMWKind(op.agg()) << ' ';
  p << op.vector() << ", ";
  p << op.memref() << '[';
  auto mapAttr = op->getAttrOfType<AffineMapAttr>("map");
  p.printAffineMapOfSSAIds(mapAttr, op.idxs());
  p << ']';
  p.printOptionalAttrDict(op->getAttrs(), {"agg", "map"});
  p << " : ";
  p.printType(op.memref().getType());
  p << ", ";
  p.printType(op.vector().getType());
}

// <operation> ::= `pxa.vector_reduce` keyword ssa-use `,` ssa-use `[`
// ssa-use-list `]`
//                 attribute-dict? `:` type
ParseResult parsePxaVectorReduceOp(OpAsmParser &parser,
                                   OperationState &result) {
  auto indexTy = parser.getBuilder().getIndexType();
  auto i64Ty = parser.getBuilder().getIntegerType(64);
  MemRefType memrefType;
  VectorType vectorType;
  AffineMapAttr mapAttr;
  OpAsmParser::OperandType val, out;
  SmallVector<OpAsmParser::OperandType, 4> idxs;
  auto symbolizeAtomicRMWKindWrap = [](StringRef str) {
    return symbolizeAtomicRMWKind(str);
  };
  return failure(
      parseKeywordIntoEnumAttr(parser, result, "agg", i64Ty,
                               symbolizeAtomicRMWKindWrap) ||
      parser.parseOperand(val) || parser.parseComma() ||
      parser.parseOperand(out) ||
      parser.parseAffineMapOfSSAIds(idxs, mapAttr, "map", result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(memrefType) ||
      parser.addTypeToList(memrefType, result.types) || parser.parseComma() ||
      parser.parseType(vectorType) ||
      parser.resolveOperand(val, vectorType, result.operands) ||
      parser.resolveOperand(out, memrefType, result.operands) ||
      parser.resolveOperands(idxs, indexTy, result.operands));
}

void PxaVectorReduceOp::getCanonicalizationPatterns(
    OwningRewritePatternList &results, MLIRContext *context) {
  results.insert<SimplifyAffineOp<PxaVectorReduceOp>,
                 SimplifyDeadReduce<PxaVectorReduceOp>>(context);
}

OpFoldResult PxaVectorReduceOp::fold(ArrayRef<Attribute> cstOperands) {
  /// vectorReduce(memrefcast) -> vectorReduce
  foldMemRefCast(*this);
  return OpFoldResult();
}

void PXADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/pxa/ir/ops.cc.inc" // NOLINT
      >();
}

} // namespace pmlc::dialect::pxa

#define GET_OP_CLASSES
#include "pmlc/dialect/pxa/ir/ops.cc.inc" // NOLINT
