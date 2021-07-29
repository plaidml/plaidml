// Copyright 2019, Intel Corporation

#include "pmlc/dialect/pxa/ir/ops.h"

#include <string>

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

#include "pmlc/util/logging.h"

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

/// Replace all occurrences of AffineExpr at position `pos` in `map` by the
/// defining AffineApplyOp expression and operands.
/// When `dimOrSymbolPosition < dims.size()`, AffineDimExpr@[pos] is replaced.
/// When `dimOrSymbolPosition >= dims.size()`,
/// AffineSymbolExpr@[pos - dims.size()] is replaced.
/// Mutate `map`,`dims` and `syms` in place as follows:
///   1. `dims` and `syms` are only appended to.
///   2. `map` dim and symbols are gradually shifted to higer positions.
///   3. Old `dim` and `sym` entries are replaced by nullptr
/// This avoids the need for any bookkeeping.
static LogicalResult replaceDimOrSym(AffineMap *map,
                                     unsigned dimOrSymbolPosition,
                                     SmallVectorImpl<Value> &dims,
                                     SmallVectorImpl<Value> &syms) {
  bool isDimReplacement = (dimOrSymbolPosition < dims.size());
  unsigned pos = isDimReplacement ? dimOrSymbolPosition
                                  : dimOrSymbolPosition - dims.size();
  Value &v = isDimReplacement ? dims[pos] : syms[pos];
  if (!v)
    return failure();

  auto affineApply = v.getDefiningOp<AffineApplyOp>();
  if (!affineApply)
    return failure();

  // At this point we will perform a replacement of `v`, set the entry in `dim`
  // or `sym` to nullptr immediately.
  v = nullptr;

  // Compute the map, dims and symbols coming from the AffineApplyOp.
  AffineMap composeMap = affineApply.getAffineMap();
  assert(composeMap.getNumResults() == 1 && "affine.apply with >1 results");
  AffineExpr composeExpr =
      composeMap.shiftDims(dims.size()).shiftSymbols(syms.size()).getResult(0);
  ValueRange composeDims =
      affineApply.getMapOperands().take_front(composeMap.getNumDims());
  ValueRange composeSyms =
      affineApply.getMapOperands().take_back(composeMap.getNumSymbols());

  // Perform the replacement and append the dims and symbols where relevant.
  MLIRContext *ctx = map->getContext();
  AffineExpr toReplace = isDimReplacement ? getAffineDimExpr(pos, ctx)
                                          : getAffineSymbolExpr(pos, ctx);
  *map = map->replace(toReplace, composeExpr, dims.size(), syms.size());
  dims.append(composeDims.begin(), composeDims.end());
  syms.append(composeSyms.begin(), composeSyms.end());

  return success();
}

/// Iterate over `operands` and fold away all those produced by an AffineApplyOp
/// iteratively. Perform canonicalization of map and operands as well as
/// AffineMap simplification. `map` and `operands` are mutated in place.
static void composeAffineMapAndOperands(AffineMap *map,
                                        SmallVectorImpl<Value> *operands) {
  if (map->getNumResults() == 0) {
    canonicalizeMapAndOperands(map, operands);
    *map = simplifyAffineMap(*map);
    return;
  }

  MLIRContext *ctx = map->getContext();
  SmallVector<Value, 4> dims(operands->begin(),
                             operands->begin() + map->getNumDims());
  SmallVector<Value, 4> syms(operands->begin() + map->getNumDims(),
                             operands->end());

  // Iterate over dims and symbols coming from AffineApplyOp and replace until
  // exhaustion. This iteratively mutates `map`, `dims` and `syms`. Both `dims`
  // and `syms` can only increase by construction.
  // The implementation uses a `while` loop to support the case of symbols
  // that may be constructed from dims ;this may be overkill.
  while (true) {
    bool changed = false;
    for (unsigned pos = 0; pos != dims.size() + syms.size(); ++pos)
      if ((changed |= succeeded(replaceDimOrSym(map, pos, dims, syms))))
        break;
    if (!changed)
      break;
  }

  // Clear operands so we can fill them anew.
  operands->clear();

  // At this point we may have introduced null operands, prune them out before
  // canonicalizing map and operands.
  unsigned nDims = 0, nSyms = 0;
  SmallVector<AffineExpr, 4> dimReplacements, symReplacements;
  dimReplacements.reserve(dims.size());
  symReplacements.reserve(syms.size());
  for (auto *container : {&dims, &syms}) {
    bool isDim = (container == &dims);
    auto &repls = isDim ? dimReplacements : symReplacements;
    for (auto en : llvm::enumerate(*container)) {
      Value v = en.value();
      if (!v) {
        assert(isDim ? !map->isFunctionOfDim(en.index())
                     : !map->isFunctionOfSymbol(en.index()) &&
                           "map is function of unexpected expr@pos");
        repls.push_back(getAffineConstantExpr(0, ctx));
        continue;
      }
      repls.push_back(isDim ? getAffineDimExpr(nDims++, ctx)
                            : getAffineSymbolExpr(nSyms++, ctx));
      operands->push_back(v);
    }
  }
  *map = map->replaceDimsAndSymbols(dimReplacements, symReplacements, nDims,
                                    nSyms);

  // Canonicalize and simplify before returning.
  canonicalizeMapAndOperands(map, operands);
  *map = simplifyAffineMap(*map);
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
                                                 op.agg(), op.val(),
                                                 op.memref(), map, mapOperands);
}

/// This is a common class used for patterns of the form
/// "someop(memrefcast) -> someop".  It folds the source of any memref_cast
/// into the root operation directly.
static LogicalResult foldMemRefCast(Operation *op) {
  bool folded = false;
  for (OpOperand &operand : op->getOpOperands()) {
    auto cast = dyn_cast_or_null<memref::CastOp>(operand.get().getDefiningOp());
    if (cast && !cast.getOperand().getType().isa<UnrankedMemRefType>()) {
      operand.set(cast.getOperand());
      folded = true;
    }
  }
  return success(folded);
}

/// Fold reduce/store operations with no uses. Reduce/store have side effects
/// on the heap, but can still be deleted if it has zero uses.
template <typename WriteOp>
struct SimplifyDeadWrite : public OpRewritePattern<WriteOp> {
  using OpRewritePattern<WriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WriteOp write,
                                PatternRewriter &rewriter) const override {
    if (write.use_empty()) {
      rewriter.eraseOp(write);
      return success();
    }
    return failure();
  }
};

struct SimplifyPxaGenericOp : public OpRewritePattern<PxaGenericOp> {
  using OpRewritePattern<PxaGenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PxaGenericOp op,
                                PatternRewriter &rewriter) const override {
    bool needsRewrite = false;

    SmallVector<Value> inputIndices;
    SmallVector<AffineMap> inputAccessMaps;
    SmallVector<AffineValueMap> inputValueMaps;
    inputValueMaps.reserve(op.getNumInputs());
    op.getAffineValueMaps(op.inputAccessMaps(), op.inputIndices(),
                          inputValueMaps);

    for (AffineValueMap &valueMap : inputValueMaps) {
      if (succeeded(valueMap.canonicalize()))
        needsRewrite = true;
      inputAccessMaps.push_back(valueMap.getAffineMap());
      inputIndices.append(valueMap.getOperands().begin(),
                          valueMap.getOperands().end());
    }

    SmallVector<Value> outputIndices;
    SmallVector<AffineMap> outputAccessMaps;
    SmallVector<AffineValueMap> outputValueMaps;
    outputValueMaps.reserve(op.getNumOutputs());
    op.getAffineValueMaps(op.outputAccessMaps(), op.outputIndices(),
                          outputValueMaps);

    for (AffineValueMap &valueMap : outputValueMaps) {
      if (succeeded(valueMap.canonicalize()))
        needsRewrite = true;
      outputAccessMaps.push_back(valueMap.getAffineMap());
      outputIndices.append(valueMap.getOperands().begin(),
                           valueMap.getOperands().end());
    }

    if (!needsRewrite)
      return failure();

    rewriter.replaceOpWithNewOp<pxa::PxaGenericOp>(
        op, op.outputs().getTypes(),
        /*inputs=*/op.inputs(),
        /*outputs=*/op.outputs(),
        /*inputIndices=*/inputIndices,
        /*outputIndices=*/outputIndices,
        /*inputAccessMaps=*/rewriter.getAffineMapArrayAttr(inputAccessMaps),
        /*inputTileMaps=*/op.inputTileMaps(),
        /*outputAccessMaps=*/rewriter.getAffineMapArrayAttr(outputAccessMaps),
        /*outputTileMaps=*/op.outputTileMaps(),
        /*kernel=*/op.kernel(),
        /*tile=*/op.tile(),
        /*reductions=*/op.reductions());

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
  p << op->getName() << ' ';
  p << op.getMemRef() << '[';
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

void PxaLoadOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                            MLIRContext *context) {
  patterns.insert<SimplifyAffineOp<PxaLoadOp>>(patterns.getContext());
}

OpFoldResult PxaLoadOp::fold(ArrayRef<Attribute> cstOperands) {
  /// load(memrefcast) -> load
  if (succeeded(foldMemRefCast(*this)))
    return getResult();
  return {};
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
  p << op->getName() << ' ';
  p << op.getMemRef() << '[';
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

void PxaVectorLoadOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *context) {
  patterns.insert<SimplifyAffineOp<PxaVectorLoadOp>>(patterns.getContext());
}

OpFoldResult PxaVectorLoadOp::fold(ArrayRef<Attribute> cstOperands) {
  /// reduce(memrefcast) -> reduce
  if (succeeded(foldMemRefCast(*this)))
    return getResult();
  return {};
}

// ---- PxaReduceOp ----

void printPxaReduceOp(OpAsmPrinter &p, PxaReduceOp op) {
  p << op->getName() << ' ';
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
  IndexType indexTy = parser.getBuilder().getIndexType();
  IntegerType i64Ty = parser.getBuilder().getIntegerType(64);
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

void PxaReduceOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *context) {
  patterns.insert<                   //
      SimplifyAffineOp<PxaReduceOp>, //
      SimplifyDeadWrite<PxaReduceOp>>(patterns.getContext());
}

OpFoldResult PxaReduceOp::fold(ArrayRef<Attribute> cstOperands) {
  /// reduce(memrefcast) -> reduce
  if (succeeded(foldMemRefCast(*this)))
    return getResult();
  return {};
}

// ---- PxaVectorReduceOp ----

void printPxaVectorReduceOp(OpAsmPrinter &p, PxaVectorReduceOp op) {
  p << op->getName() << ' ';
  p << stringifyAtomicRMWKind(op.agg()) << ' ';
  p << op.val() << ", ";
  p << op.memref() << '[';
  auto mapAttr = op->getAttrOfType<AffineMapAttr>("map");
  p.printAffineMapOfSSAIds(mapAttr, op.idxs());
  p << ']';
  p.printOptionalAttrDict(op->getAttrs(), {"agg", "map"});
  p << " : ";
  p.printType(op.memref().getType());
  p << ", ";
  p.printType(op.val().getType());
}

// <operation> ::= `pxa.vector_reduce` keyword ssa-use `,` ssa-use `[`
//                 ssa-use-list `]` attribute-dict? `:` type
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

void PxaVectorReduceOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                    MLIRContext *context) {
  patterns.insert<SimplifyAffineOp<PxaVectorReduceOp>,
                  SimplifyDeadWrite<PxaVectorReduceOp>>(patterns.getContext());
}

OpFoldResult PxaVectorReduceOp::fold(ArrayRef<Attribute> cstOperands) {
  /// vectorReduce(memrefcast) -> vectorReduce
  if (succeeded(foldMemRefCast(*this)))
    return getResult();
  return {};
}

// ---- PxaStoreOp ----

void PxaStoreOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                             MLIRContext *context) {
  patterns.insert<SimplifyDeadWrite<PxaStoreOp>>(patterns.getContext());
}

OpFoldResult PxaStoreOp::fold(ArrayRef<Attribute> cstOperands) {
  /// store(memrefcast) -> store
  if (succeeded(foldMemRefCast(*this)))
    return getResult();
  return {};
}

void PXADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/pxa/ir/ops.cc.inc" // NOLINT
      >();
}

//
// ---- PxaGenericOp ----
//

bool PxaGenericOp::isReadOperand(mlir::OpOperand *opOperand) {
  assert(opOperand->getOwner() == this->getOperation());
  return opOperand->getOperandNumber() < getNumInputs();
}

bool PxaGenericOp::isWriteOperand(mlir::OpOperand *opOperand) {
  assert(opOperand->getOwner() == this->getOperation());
  unsigned operandNumber = opOperand->getOperandNumber();
  return operandNumber >= getNumInputs() &&
         operandNumber < getNumInputs() + getNumOutputs();
}

OpOperandVector PxaGenericOp::getInputOperands() {
  int64_t numInputs = getNumInputs();
  OpOperandVector result;
  result.reserve(numInputs);
  for (OpOperand &opOperand :
       getOperation()->getOpOperands().take_front(numInputs)) {
    result.push_back(&opOperand);
  }
  return result;
}

OpOperandVector PxaGenericOp::getOutputOperands() {
  int64_t numOutputs = getNumOutputs();
  OpOperandVector result;
  result.reserve(numOutputs);
  MutableArrayRef<OpOperand> opOperands = getOperation()->getOpOperands();
  for (OpOperand &opOperand :
       opOperands.drop_front(getNumInputs()).take_front(numOutputs)) {
    result.push_back(&opOperand);
  }
  return result;
}

static Optional<StrideRange> computeRange(AffineExpr expr,
                                          ArrayRef<int64_t> values) {
  // If we are a constant affine expression, the range is a fixed value
  if (auto cexpr = expr.dyn_cast<AffineConstantExpr>())
    return StrideRange(cexpr.getValue());

  // If we are a dim, the range is [0, value - 1]
  if (auto dexpr = expr.dyn_cast<AffineDimExpr>())
    return StrideRange(0, values[dexpr.getPosition()] - 1, 1);

  // Check the various binary ops.
  if (auto bexpr = expr.dyn_cast<AffineBinaryOpExpr>()) {
    if (bexpr.getKind() == AffineExprKind::Mul) {
      // For multiplies, RHS should always be constant of symbolic, and symbols
      // fail, so we cast to constant and give up if it doesn't work
      auto rhs = bexpr.getRHS().dyn_cast<AffineConstantExpr>();
      if (!rhs)
        return None;

      // Now compute the LHS via recursion
      Optional<StrideRange> lhs = computeRange(bexpr.getLHS(), values);
      if (!lhs)
        return None;

      // Multiply by the multiplier and return
      *lhs *= rhs.getValue();
      return lhs;
    }

    if (bexpr.getKind() == AffineExprKind::Add) {
      // For add, we compute both sides and add them (presuming they both return
      // valid outputs).
      Optional<StrideRange> lhs = computeRange(bexpr.getLHS(), values);
      if (!lhs)
        return None;

      Optional<StrideRange> rhs = computeRange(bexpr.getRHS(), values);
      if (!rhs)
        return None;

      *lhs += *rhs;
      return lhs;
    }
  }

  return None;
}

SmallVector<StrideRange>
PxaGenericOp::getTiedInternalRanges(OpOperand *opOperand) {
  unsigned operandNumber = opOperand->getOperandNumber();
  assert(opOperand->getOwner() == this->getOperation());
  assert(operandNumber < getNumInputs() + getNumOutputs());

  Attribute attr;
  if (operandNumber < getNumInputs())
    attr = inputTileMaps()[operandNumber];
  else
    attr = outputTileMaps()[operandNumber - getNumInputs()];
  AffineMap tileMap = attr.cast<AffineMapAttr>().getValue();

  SmallVector<int64_t> tileSizes;
  for (APInt value : tile().getAsValueRange<IntegerAttr>())
    tileSizes.push_back(value.getZExtValue());

  SmallVector<StrideRange> results;
  for (AffineExpr expr : tileMap.getResults()) {
    Optional<StrideRange> range = computeRange(expr, tileSizes);
    assert(range.hasValue() && "Invalid StrideRange in tileMap");
    results.push_back(*range);
  }

  return results;
}

void PxaGenericOp::getAffineValueMaps(
    ArrayAttr accessMaps, OperandRange mapOperands,
    SmallVectorImpl<AffineValueMap> &results) {
  size_t prefix = 0;
  for (Attribute accessMapAttr : accessMaps) {
    AffineMap accessMap = accessMapAttr.cast<AffineMapAttr>().getValue();
    size_t count = accessMap.getNumInputs();
    AffineValueMap valueMap(accessMap, mapOperands.slice(prefix, count));
    results.emplace_back(valueMap);
    prefix += count;
  }
}

SmallVector<AffineValueMap> PxaGenericOp::getAffineValueMaps() {
  SmallVector<AffineValueMap> result;
  result.reserve(getNumInputs() + getNumOutputs());
  getAffineValueMaps(inputAccessMaps(), inputIndices(), result);
  getAffineValueMaps(outputAccessMaps(), outputIndices(), result);
  return result;
}

AffineValueMap PxaGenericOp::getTiedAffineValueMap(OpOperand *opOperand) {
  assert(opOperand->getOwner() == this->getOperation());
  assert(opOperand->getOperandNumber() < getNumInputs() + getNumOutputs());
  return getAffineValueMaps()[opOperand->getOperandNumber()];
}

void PxaGenericOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                               MLIRContext *context) {
  patterns.insert<SimplifyPxaGenericOp>(patterns.getContext());
}

static void printPxaGenericOperands(OpAsmPrinter &p, OperandRange operands,
                                    OperandRange indices,
                                    ArrayAttr accessMapsAttr,
                                    ArrayAttr tileMapsAttr) {
  auto items = llvm::zip(operands, accessMapsAttr, tileMapsAttr);
  size_t prefix = 0;
  llvm::interleaveComma(items, p, [&](auto it) {
    Value operand;
    Attribute accessMap, tileMap;
    std::tie(operand, accessMap, tileMap) = it;
    p << operand << '[';
    AffineMapAttr accessMapAttr = accessMap.cast<AffineMapAttr>();
    size_t count = accessMapAttr.getValue().getNumInputs();
    p.printAffineMapOfSSAIds(accessMapAttr, indices.slice(prefix, count));
    prefix += count;
    p << "]: " << tileMap;
  });
}

static void printPxaGenericOp(OpAsmPrinter &p, PxaGenericOp op) {
  auto funcType = FunctionType::get(op.getContext(), op.inputs().getTypes(),
                                    op.outputs().getTypes());
  p << op->getName() << ' ';
  p << '(';
  printPxaGenericOperands(p, op.outputs(), op.outputIndices(),
                          op.outputAccessMaps(), op.outputTileMaps());
  p << ") <";
  llvm::interleaveComma(op.reductions(), p, [&](Attribute attr) {
    Optional<AtomicRMWKind> kind =
        symbolizeAtomicRMWKind(attr.cast<IntegerAttr>().getInt());
    p << stringifyAtomicRMWKind(*kind);
  });
  p << "> ";
  p.printSymbolName(op.kernel());
  p << '(';
  printPxaGenericOperands(p, op.inputs(), op.inputIndices(),
                          op.inputAccessMaps(), op.inputTileMaps());
  p << ") tile: " << op.tile();
  p.printOptionalAttrDict(op->getAttrs(),
                          /*elidedAttrs=*/{
                              PxaGenericOp::getOperandSegmentSizeAttr(),
                              PxaGenericOp::getInputAccessMapsAttrName(),
                              PxaGenericOp::getInputTileMapsAttrName(),
                              PxaGenericOp::getOutputAccessMapsAttrName(),
                              PxaGenericOp::getOutputTileMapsAttrName(),
                              PxaGenericOp::getKernelAttrName(),
                              PxaGenericOp::getTileAttrName(),
                              PxaGenericOp::getReductionsAttrName(),
                          });
  p << " : " << funcType;
}

struct GenericOperand {
  OpAsmParser::OperandType operand;
  SmallVector<OpAsmParser::OperandType, 4> indices;
  AffineMapAttr accessMapAttr;
  AffineMapAttr tileMapAttr;

  ParseResult parse(OpAsmParser &parser) {
    NamedAttrList attrs;
    return failure(parser.parseOperand(operand) ||
                   parser.parseAffineMapOfSSAIds(indices, accessMapAttr,
                                                 "accessMapAttr", attrs) ||
                   parser.parseColon() ||
                   parser.parseAttribute(tileMapAttr, "tileMapAttr", attrs));
  }
};

struct GenericOperands {
  SmallVector<OpAsmParser::OperandType> operands;
  SmallVector<OpAsmParser::OperandType> indices;

  ParseResult parse(OpAsmParser &parser, StringRef accessMapAttrName,
                    StringRef tileMapAttrName, OperationState &result) {
    if (failed(parser.parseLParen()))
      return failure();

    Builder &builder = parser.getBuilder();
    SmallVector<Attribute> accessMapAttrs, tileMapAttrs;

    do {
      GenericOperand operand;
      if (failed(operand.parse(parser)))
        return failure();
      operands.push_back(operand.operand);
      indices.append(operand.indices);
      accessMapAttrs.push_back(operand.accessMapAttr);
      tileMapAttrs.push_back(operand.tileMapAttr);
    } while (succeeded(parser.parseOptionalComma()));

    if (failed(parser.parseRParen()))
      return failure();

    result.addAttribute(accessMapAttrName,
                        builder.getArrayAttr(accessMapAttrs));
    result.addAttribute(tileMapAttrName, builder.getArrayAttr(tileMapAttrs));
    return success();
  }
};

static ParseResult parseReductions(OpAsmParser &parser,
                                   OperationState &result) {
  if (failed(parser.parseLess()))
    return failure();

  SmallVector<int64_t> reductions;
  do {
    llvm::SMLoc loc;
    StringRef str;
    if (parser.getCurrentLocation(&loc) || parser.parseKeyword(&str))
      return failure();

    Optional<AtomicRMWKind> kind = symbolizeAtomicRMWKind(str);
    if (!kind)
      return parser.emitError(loc) << "expected valid AtomicRMWKind value";

    reductions.push_back(static_cast<int64_t>(*kind));
  } while (succeeded(parser.parseOptionalComma()));

  if (failed(parser.parseGreater()))
    return failure();

  Builder &builder = parser.getBuilder();
  result.addAttribute(PxaGenericOp::getReductionsAttrName(),
                      builder.getI64ArrayAttr(reductions));
  return success();
}

static ParseResult parsePxaGenericOp(OpAsmParser &parser,
                                     OperationState &result) {
  Builder &builder = parser.getBuilder();
  IndexType indexType = builder.getIndexType();
  IntegerType i64Type = builder.getIntegerType(64);
  GenericOperands inputs, outputs;
  FunctionType funcType;
  StringAttr kernel;
  ArrayAttr tileAttr;
  if (outputs.parse(parser, PxaGenericOp::getOutputAccessMapsAttrName(),
                    PxaGenericOp::getOutputTileMapsAttrName(), result) ||
      parseReductions(parser, result) ||
      parser.parseSymbolName(kernel, PxaGenericOp::getKernelAttrName(),
                             result.attributes) ||
      inputs.parse(parser, PxaGenericOp::getInputAccessMapsAttrName(),
                   PxaGenericOp::getInputTileMapsAttrName(), result) ||
      parser.parseKeyword("tile") || parser.parseColon() ||
      parser.parseAttribute(tileAttr, i64Type, PxaGenericOp::getTileAttrName(),
                            result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(funcType) ||
      parser.resolveOperands(inputs.operands, funcType.getInputs(),
                             parser.getNameLoc(), result.operands) ||
      parser.resolveOperands(outputs.operands, funcType.getResults(),
                             parser.getNameLoc(), result.operands) ||
      parser.resolveOperands(inputs.indices, indexType, result.operands) ||
      parser.resolveOperands(outputs.indices, indexType, result.operands) ||
      parser.addTypesToList(funcType.getResults(), result.types))
    return failure();

  result.addAttribute(PxaGenericOp::getOperandSegmentSizeAttr(),
                      builder.getI32VectorAttr({
                          static_cast<int32_t>(inputs.operands.size()),
                          static_cast<int32_t>(outputs.operands.size()),
                          static_cast<int32_t>(inputs.indices.size()),
                          static_cast<int32_t>(outputs.indices.size()),
                      }));
  return success();
}

} // namespace pmlc::dialect::pxa

#include "pmlc/dialect/pxa/ir/dialect.cc.inc" // NOLINT

#define GET_OP_CLASSES
#include "pmlc/dialect/pxa/ir/ops.cc.inc" // NOLINT
