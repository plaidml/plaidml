// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/ir/ops.h"

#include <vector>

#include "llvm/ADT/SetVector.h"

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/dialect/eltwise/ir/util.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/util.h"

namespace pmlc::dialect::tile {

using eltwise::constFoldBinaryOp;
using eltwise::constFoldUnaryOp;
using eltwise::m_One;
using eltwise::m_Zero;
using llvm::SetVector;
using llvm::SmallVector;
using mlir::AffineExpr;
using mlir::ArrayAttr;
using mlir::failure;
using mlir::FloatAttr;
using mlir::IntegerAttr;
using mlir::OpRewritePattern;
using mlir::PatternMatchResult;
using mlir::PatternRewriter;
using mlir::StringAttr;
using mlir::success;
using mlir::Value;

OpFoldResult AffineConstantOp::fold(ArrayRef<Attribute> operands) {
  IVLOG(5, "AffineConstantOp::fold");
  assert(operands.empty() && "constant has no operands");
  return getValue();
}

OpFoldResult AffineAddOp::fold(ArrayRef<Attribute> operands) {
  IVLOG(5, "AffineAddOp::fold");
  /// add(x, 0) -> x
  if (matchPattern(rhs(), m_Zero())) {
    return lhs();
  }
  return constFoldBinaryOp(operands, [](double a, double b) { return a + b; });
}

OpFoldResult AffineDivOp::fold(ArrayRef<Attribute> operands) {
  IVLOG(5, "AffineDivOp::fold");
  // Don't fold if it requires division by zero.
  if (matchPattern(rhs(), m_Zero())) {
    return {};
  }
  // div(x, 1) -> x
  if (matchPattern(rhs(), m_One())) {
    return lhs();
  }
  // div(0, x) -> 0
  if (matchPattern(lhs(), m_Zero())) {
    Builder builder(getContext());
    return builder.getZeroAttr(builder.getIntegerType(64));
  }
  return constFoldBinaryOp(operands, [](double a, double b) { return a / b; });
}

OpFoldResult AffineMulOp::fold(ArrayRef<Attribute> operands) {
  IVLOG(5, "AffineMulOp::fold");
  // mul(x, 0) -> 0
  if (matchPattern(rhs(), m_Zero())) {
    IVLOG(5, "mul(x, 0) -> 0");
    return rhs();
  }
  // mul(x, 1) -> x
  if (matchPattern(rhs(), m_One())) {
    IVLOG(5, "mul(x, 1) -> x");
    return lhs();
  }
  return constFoldBinaryOp(operands, [](double a, double b) {
    IVLOG(5, a << " * " << b << " = " << a * b);
    return a * b;
  });
}

OpFoldResult AffineNegOp::fold(ArrayRef<Attribute> operands) {
  IVLOG(5, "AffineNegOp::fold");
  return constFoldUnaryOp(operands, [](double x) { return -x; });
}

OpFoldResult AffineMaxOp::fold(ArrayRef<Attribute> operands) {
  IVLOG(5, "AffineMaxOp::fold");
  return constFoldBinaryOp(operands, [](double a, double b) { return fmax(a, b); });
}

OpFoldResult AffineMinOp::fold(ArrayRef<Attribute> operands) {
  IVLOG(5, "AffineMinOp::fold");
  return constFoldBinaryOp(operands, [](double a, double b) { return fmin(a, b); });
}

OpFoldResult AffineSubOp::fold(ArrayRef<Attribute> operands) {
  IVLOG(5, "AffineSubOp::fold");
  // sub(x, x) -> 0
  if (lhs() == rhs()) {
    IVLOG(5, "sub(x, x) -> 0");
    Builder builder(getContext());
    return builder.getZeroAttr(builder.getIntegerType(64));
  }
  /// sub(x, 0) -> x
  if (matchPattern(rhs(), m_Zero())) {
    IVLOG(5, "sub(x, 0) -> x");
    return lhs();
  }
  return constFoldBinaryOp(operands, [](double a, double b) { return a - b; });
}

OpFoldResult DimOp::fold(ArrayRef<Attribute> operands) {
  IVLOG(5, "DimOp::fold");
  return resolve();
}

template <typename T, typename R = void>
class AffineVisitor {
 public:
  R visit(Value value) {
    static_assert(std::is_base_of<AffineVisitor, T>::value, "Must instantiate with a derived type of AffineVisitor");
    auto defOp = value->getDefiningOp();
    if (auto op = llvm::dyn_cast<AffineIndexOp>(defOp)) {
      return static_cast<T*>(this)->visitIndexOp(op);
    } else if (auto op = llvm::dyn_cast<AffineConstantOp>(defOp)) {
      return static_cast<T*>(this)->visitConstantOp(op);
    } else if (auto op = llvm::dyn_cast<AffineAddOp>(defOp)) {
      visitOperands(op.lhs(), op.rhs());
      return static_cast<T*>(this)->visitAddOp(op);
    } else if (auto op = llvm::dyn_cast<AffineMulOp>(defOp)) {
      visitOperands(op.lhs(), op.rhs());
      return static_cast<T*>(this)->visitMulOp(op);
    } else if (auto op = llvm::dyn_cast<AffineDivOp>(defOp)) {
      visitOperands(op.lhs(), op.rhs());
      return static_cast<T*>(this)->visitDivOp(op);
    } else if (auto op = llvm::dyn_cast<AffineNegOp>(defOp)) {
      visit(op.input());
      return static_cast<T*>(this)->visitNegOp(op);
    } else if (auto op = llvm::dyn_cast<AffineSubOp>(defOp)) {
      visitOperands(op.lhs(), op.rhs());
      return static_cast<T*>(this)->visitSubOp(op);
    } else if (auto op = llvm::dyn_cast<AffineMaxOp>(defOp)) {
      visitOperands(op.lhs(), op.rhs());
      return static_cast<T*>(this)->visitMaxOp(op);
    } else if (auto op = llvm::dyn_cast<AffineMinOp>(defOp)) {
      visitOperands(op.lhs(), op.rhs());
      return static_cast<T*>(this)->visitMinOp(op);
    } else if (auto op = llvm::dyn_cast<DimOp>(defOp)) {
      return static_cast<T*>(this)->visitDimOp(op);
    }
    llvm_unreachable("Unknown Affine op");
  }

  void visitIndexOp(AffineIndexOp op) {}
  void visitConstantOp(AffineConstantOp op) {}
  void visitAddOp(AffineAddOp op) {}
  void visitMulOp(AffineMulOp op) {}
  void visitDivOp(AffineDivOp op) {}
  void visitSubOp(AffineSubOp op) {}
  void visitNegOp(AffineNegOp op) {}
  void visitMaxOp(AffineMaxOp op) {}
  void visitMinOp(AffineMinOp op) {}
  void visitDimOp(DimOp op) {}

 private:
  void visitOperands(Value lhs, Value rhs) {
    visit(lhs);
    visit(rhs);
  }
};

struct AffineIndexCollector : public AffineVisitor<AffineIndexCollector> {
  SetVector<Value> idxs;
  void visitIndexOp(AffineIndexOp op) { idxs.insert(op.result()); }
};

struct IsFoldableVisitor : public AffineVisitor<IsFoldableVisitor> {
  bool foldable = true;
  void visitMaxOp(AffineMaxOp op) { foldable = false; }
  void visitMinOp(AffineMinOp op) { foldable = false; }

  bool is_foldable(SymbolicContractionOp op) {
    foldable = true;
    auto sinkMapOp = llvm::cast<AffineMapOp>(op.sink()->getDefiningOp());
    for (auto dim : sinkMapOp.dims()) {
      visit(dim);
    }

    auto sizeMapOp = llvm::cast<AffineMapOp>(op.size()->getDefiningOp());
    for (auto dim : sizeMapOp.dims()) {
      visit(dim);
    }

    for (auto src : op.srcs()) {
      auto mapOp = llvm::cast<AffineTensorMapOp>(src->getDefiningOp());
      for (auto dim : mapOp.dims()) {
        visit(dim);
      }
    }

    auto consOp = llvm::cast<AffineConstraintsOp>(op.cons()->getDefiningOp());
    for (auto pair : consOp.pairs()) {
      visit(pair);
    }
    return foldable;
  }
};

struct ContractionBuilder : public AffineVisitor<ContractionBuilder, AffineExpr> {
  friend class AffineVisitor<ContractionBuilder, AffineExpr>;

 public:
  explicit ContractionBuilder(SymbolicContractionOp op) : context(op.getContext()) {
    auto sinkMapOp = llvm::cast<AffineMapOp>(op.sink()->getDefiningOp());
    auto consOp = llvm::cast<AffineConstraintsOp>(op.cons()->getDefiningOp());

    // first collect all the indexes
    for (auto dim : sinkMapOp.dims()) {
      collector.visit(dim);
    }

    for (auto src : op.srcs()) {
      auto mapOp = llvm::cast<AffineTensorMapOp>(src->getDefiningOp());
      for (auto dim : mapOp.dims()) {
        collector.visit(dim);
      }
    }

    for (auto pair : consOp.pairs()) {
      collector.visit(pair);
    }

    // now construct the AffineExprs
    sink = addDims(sinkMapOp.dims());

    for (auto src : op.srcs()) {
      auto mapOp = llvm::cast<AffineTensorMapOp>(src->getDefiningOp());
      addSourceMap(mapOp);
    }

    SmallVector<Value, 8> pairs(consOp.pairs());
    for (auto it = pairs.begin(); it != pairs.end(); it++) {
      auto lhs = *it++;
      auto rhs = *it;
      cons.emplace_back(mlir::simplifyAffineExpr(makeExpr(lhs), collector.idxs.size(), 0));
      cons.emplace_back(makeConstraint(lhs, rhs));
    }
  }

  std::vector<AffineMap> getSources() {
    std::vector<AffineMap> ret;
    for (auto src : srcs) {
      ret.emplace_back(makeMap(src));
    }
    return ret;
  }

  AffineMap getSink() {  //
    return makeMap(sink);
  }

  IntegerSet getConstraints() {
    if (cons.empty()) {
      return IntegerSet::getEmptySet(collector.idxs.size(), 0, context);
    }
    SmallVector<bool, 4> flags(cons.size(), false);
    return IntegerSet::get(collector.idxs.size(), 0, cons, flags);
  }

  ArrayRef<Value> getIndexes() {  //
    return collector.idxs.getArrayRef();
  }

  ArrayRef<Value> getTensors() {  //
    return llvm::makeArrayRef(tensors);
  }

 private:
  MLIRContext* context;
  AffineIndexCollector collector;
  std::vector<Value> tensors;
  std::vector<AffineExpr> sink;
  std::vector<std::vector<AffineExpr>> srcs;
  std::vector<AffineExpr> cons;

  std::vector<AffineExpr> addDims(Operation::operand_range dims) {
    std::vector<AffineExpr> exprs;
    for (auto dim : dims) {
      exprs.emplace_back(makeExpr(dim));
    }
    return exprs;
  }

  void addSourceMap(AffineTensorMapOp op) {
    tensors.emplace_back(op.tensor());
    srcs.emplace_back(addDims(op.dims()));
  }

  AffineMap makeMap(ArrayRef<AffineExpr> exprs) {
    if (exprs.size()) {
      return AffineMap::get(collector.idxs.size(), 0, exprs);
    }
    return AffineMap::get(context);
  }

  AffineExpr makeConstraint(Value lhs, Value rhs) {
    // Constraints are in the form `lhs < rhs`
    // IntegerSets are in the form `x >= 0`
    // lhs < rhs
    // -lhs > -rhs             multiply -1
    // rhs - lhs > 0           add rhs
    // rhs - lhs - 1 >= 0      subtract 1
    auto expr = makeExpr(rhs) - makeExpr(lhs) - 1;
    return mlir::simplifyAffineExpr(expr, collector.idxs.size(), 0);
  }

  AffineExpr makeExpr(Value value) {  //
    return visit(value);
  }

  AffineExpr visitIndexOp(AffineIndexOp op) {  //
    auto value = op.result();
    auto pos = std::distance(collector.idxs.begin(), llvm::find(collector.idxs, value));
    return mlir::getAffineDimExpr(pos, context);
  }

  AffineExpr visitConstantOp(AffineConstantOp op) {  //
    return mlir::getAffineConstantExpr(op.value().getSExtValue(), context);
  }

  AffineExpr visitAddOp(AffineAddOp op) {  //
    return makeExpr(op.lhs()) + makeExpr(op.rhs());
  }

  AffineExpr visitMulOp(AffineMulOp op) {  //
    return makeExpr(op.lhs()) * makeExpr(op.rhs());
  }

  AffineExpr visitSubOp(AffineSubOp op) {  //
    return makeExpr(op.lhs()) - makeExpr(op.rhs());
  }

  AffineExpr visitDivOp(AffineDivOp op) {  //
    return makeExpr(op.lhs()).floorDiv(makeExpr(op.rhs()));
  }

  AffineExpr visitNegOp(AffineNegOp op) {  //
    return -makeExpr(op.input());
  }

  AffineExpr visitMaxOp(AffineMaxOp op) {  //
    llvm_unreachable("Max op not legal in ContractionBuilder affines, this should have been folded earlier");
  }

  AffineExpr visitMinOp(AffineMinOp op) {  //
    llvm_unreachable("Min op not legal in ContractionBuilder affines, this should have been folded earlier");
  }

  AffineExpr visitDimOp(DimOp op) {  //
    if (auto attr = op.resolve()) {
      return mlir::getAffineConstantExpr(attr.getInt(), context);
    }
    llvm_unreachable("Invalid DimOp, must resolve to constant value");
  }
};

struct SymbolicContractionCanonicalizer : OpRewritePattern<SymbolicContractionOp> {
  using OpRewritePattern<SymbolicContractionOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(SymbolicContractionOp op, PatternRewriter& rewriter) const override {
    auto sizeMapOp = llvm::cast<AffineMapOp>(op.size()->getDefiningOp());
    SmallVector<Value, 4> sizeDims(sizeMapOp.dims());
    auto shape = eltwise::ComputeShape(sizeDims);
    auto sourceType = op.result()->getType().cast<RankedTensorType>();
    auto resultType = RankedTensorType::get(shape, sourceType.getElementType());
    if (!resultType.hasStaticShape()) {
      return matchFailure();
    }
    IsFoldableVisitor foldable_checker;
    if (!foldable_checker.is_foldable(op)) {
      return matchFailure();
    }

    ContractionBuilder builder(op);
    auto newOp = rewriter.create<ContractionOp>(  //
        op.getLoc(),                              //
        resultType,                               //
        op.init(),                                //
        builder.getTensors(),                     //
        op.agg(),                                 //
        op.combo(),                               //
        builder.getSink(),                        //
        builder.getSources(),                     //
        builder.getConstraints(),                 //
        op.no_reduce().hasValue(),                //
        op.name().getValueOr(""));
    bool hasNames = false;
    auto idxs = builder.getIndexes();
    SmallVector<Attribute, 8> idxNames;
    for (unsigned i = 0; i < idxs.size(); i++) {
      auto idx = idxs[i];
      auto indexOp = llvm::cast<AffineIndexOp>(idx->getDefiningOp());
      if (auto attr = indexOp.getAttrOfType<StringAttr>("name")) {
        idxNames.emplace_back(attr);
        hasNames = true;
      } else {
        auto name = llvm::formatv("x{0}", i);
        idxNames.emplace_back(rewriter.getStringAttr(name.str()));
      }
    }
    if (hasNames) {
      newOp.setAttr("idxs", ArrayAttr::get(idxNames, rewriter.getContext()));
    }
    rewriter.replaceOp(op, newOp.result());

    util::UpdateFuncOpType(newOp.getOperation());

    return matchSuccess();
  }
};

void SymbolicContractionOp::getCanonicalizationPatterns(  //
    OwningRewritePatternList& results,                    //
    MLIRContext* context) {
  results.insert<SymbolicContractionCanonicalizer>(context);
}

unsigned ContractionOp::getNumTensors(CombinationKind combo) {
  switch (combo) {
    case CombinationKind::none:
      return 1;
    case CombinationKind::add:
    case CombinationKind::eq:
    case CombinationKind::mul:
      return 2;
    case CombinationKind::cond:
      return 3;
    default:
      throw std::runtime_error("Invalid combination op");
  }
}

void ContractionOp::build(     //
    Builder* builder,          //
    OperationState& result,    //
    Type resultType,           //
    Value init,                //
    ArrayRef<Value> tensors,   //
    AggregationKind agg,       //
    CombinationKind combo,     //
    AffineMap sink,            //
    ArrayRef<AffineMap> srcs,  //
    IntegerSet cons,           //
    bool no_reduce,            //
    StringRef name) {
  result.addOperands(init);
  result.addOperands(tensors);
  result.addTypes(resultType);
  result.addAttribute("agg", builder->getI64IntegerAttr(static_cast<int64_t>(agg)));
  result.addAttribute("combo", builder->getI64IntegerAttr(static_cast<int64_t>(combo)));
  result.addAttribute(getSinkAttrName(), AffineMapAttr::get(sink));
  result.addAttribute(getSourcesAttrName(), builder->getAffineMapArrayAttr(srcs));
  if (!cons.isEmptyIntegerSet()) {
    result.addAttribute(getConstraintsAttrName(), IntegerSetAttr::get(cons));
  }
  if (no_reduce) {
    result.addAttribute("no_reduce", builder->getUnitAttr());
  }
  if (name.size()) {
    result.addAttribute("name", builder->getStringAttr(name));
  }
}

void ContractionOp::setLowerBounds(ArrayRef<int64_t> bounds) {
  SmallVector<AffineExpr, 6> exprs;
  for (auto dim : bounds) {
    exprs.push_back(mlir::getAffineConstantExpr(dim, getContext()));
  }
  auto map = AffineMap::get(/*dimCount=*/0, /*symbolCount=*/0, exprs);
  setAttr(getLowerBoundsAttrName(), AffineMapAttr::get(map));
}

void ContractionOp::setUpperBounds(ArrayRef<int64_t> bounds) {
  SmallVector<AffineExpr, 6> exprs;
  for (auto dim : bounds) {
    exprs.push_back(mlir::getAffineConstantExpr(dim, getContext()));
  }
  auto map = AffineMap::get(/*dimCount=*/0, /*symbolCount=*/0, exprs);
  setAttr(getUpperBoundsAttrName(), AffineMapAttr::get(map));
}

void ContractionOp::setSink(AffineMap sink) {  //
  setAttr(getSinkAttrName(), AffineMapAttr::get(sink));
}

void ContractionOp::setSources(ArrayRef<AffineMap> srcs) {
  SmallVector<Attribute, 4> attrs;
  for (auto src : srcs) {
    attrs.push_back(AffineMapAttr::get(src));
  }
  setAttr(getSourcesAttrName(), ArrayAttr::get(attrs, getContext()));
}

void ContractionOp::setConstraints(IntegerSet cons) {
  if (cons.isEmptyIntegerSet()) {
    removeAttr(getConstraintsAttrName());
  } else {
    setAttr(getConstraintsAttrName(), IntegerSetAttr::get(cons));
  }
}

//
// --- GatherOp ---
//

struct GatherCanonicalizer : public OpRewritePattern<GatherOp> {
  using OpRewritePattern<GatherOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(GatherOp gatherOp, PatternRewriter& rewriter) const override {
    IVLOG(5, "IndexCanonicalizer::matchAndRewrite> " << mlir::debugString(gatherOp));
    auto op = gatherOp.getOperation();
    SmallVector<Value, 2> operands(op->getOperands());
    auto resultType = GatherOp::getResultType(operands);
    if (resultType == gatherOp.result()->getType()) {
      return Pattern::matchFailure();
    }
    auto newOp = rewriter.create<GatherOp>(op->getLoc(), resultType, gatherOp.tensor(), gatherOp.dims());
    rewriter.replaceOp(op, {newOp});
    util::UpdateFuncOpType(newOp.getOperation());
    return Pattern::matchSuccess();
  }
};

void GatherOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<GatherCanonicalizer>(context);
}

Type GatherOp::getResultType(ArrayRef<Value> operands) {
  IVLOG(5, "GatherOp::getResultType>")
  if (operands.size() != 2) {
    throw std::runtime_error("GatherOp requires 2 operands");
  }
  auto tensor = operands[0];
  auto tensorType = eltwise::getRankedTensorType(tensor->getType());
  auto tensorElementType = tensorType.getElementType();
  if (!tensorType.getRank()) {
    throw std::runtime_error("'gather' requires first operand to have at least one dimension.");
  }
  auto index = operands[1];
  auto indexType = eltwise::getRankedTensorType(index->getType());
  auto indexElementType = indexType.getElementType().dyn_cast<ScalarType>();
  if (!indexElementType || indexElementType.type() != eltwise::DataType::i32) {
    throw std::runtime_error("'gather' requires the data type for the second argument to be i32.");
  }
  SmallVector<int64_t, 4> shape;
  auto tensorShape = tensorType.getShape();
  auto indexShape = indexType.getShape();
  for (size_t i = 0; i < indexShape.size(); i++) {
    shape.push_back(indexShape[i]);
  }
  for (size_t i = 1; i < tensorShape.size(); i++) {
    shape.push_back(tensorShape[i]);
  }
  auto resultType = RankedTensorType::get(shape, tensorElementType);
  IVLOG(6, "  resultType: " << mlir::debugString(resultType));
  return resultType;
}

//
// ---- IndexOp ----
//

struct IndexCanonicalizer : public OpRewritePattern<IndexOp> {
  using OpRewritePattern<IndexOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(IndexOp indexOp, PatternRewriter& rewriter) const override {
    IVLOG(5, "IndexCanonicalizer::matchAndRewrite> " << mlir::debugString(indexOp));
    auto op = indexOp.getOperation();
    SmallVector<Value, 2> operands(op->getOperands());
    auto resultType = IndexOp::getResultType(operands);
    if (resultType == indexOp.result()->getType()) {
      return Pattern::matchFailure();
    }
    auto dim = indexOp.getAttrOfType<IntegerAttr>("dim");
    auto newOp = rewriter.create<IndexOp>(op->getLoc(), resultType, indexOp.tensor(), dim);
    rewriter.replaceOp(op, {newOp});
    util::UpdateFuncOpType(newOp.getOperation());
    return Pattern::matchSuccess();
  }
};

void IndexOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<IndexCanonicalizer>(context);
}

Type IndexOp::getResultType(ArrayRef<Value> operands) {
  IVLOG(5, "IndexOp::getResultType>")
  for (auto operand : operands) {
    IVLOG(6, "  operand: " << mlir::debugString(operand));
  }
  if (operands.size() != 1) {
    throw std::runtime_error("IndexOp requires 1 operand");
  }
  auto tensor = operands.front();
  auto tensorType = eltwise::getRankedTensorType(tensor->getType());
  auto elementType = ScalarType::get(tensor->getContext(), eltwise::DataType::i32);
  IVLOG(6, "  elementType: " << mlir::debugString(elementType));
  auto resultType = RankedTensorType::get(tensorType.getShape(), elementType);
  IVLOG(6, "  resultType: " << mlir::debugString(resultType));
  return resultType;
}

//
// ---- PrngOp ----
//

struct PrngCanonicalizer : public OpRewritePattern<PrngOp> {
  using OpRewritePattern<PrngOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(PrngOp prngOp, PatternRewriter& rewriter) const override {
    IVLOG(5, "PrngCanonicalizer::matchAndRewrite> " << mlir::debugString(prngOp));
    auto op = prngOp.getOperation();
    SmallVector<Value, 5> operands(op->getOperands());
    auto resultType = PrngOp::getResultType(operands);
    if (resultType == prngOp.result()->getType()) {
      return Pattern::matchFailure();
    }
    auto stateType = prngOp.new_state()->getType();
    SmallVector<Value, 4> dims(prngOp.dims());
    auto newOp = rewriter.create<PrngOp>(op->getLoc(), resultType, stateType, prngOp.state(), dims);
    rewriter.replaceOp(op, {newOp.result(), newOp.new_state()});
    util::UpdateFuncOpType(newOp.getOperation());
    return Pattern::matchSuccess();
  }
};

void PrngOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<PrngCanonicalizer>(context);
}

Type PrngOp::getResultType(ArrayRef<Value> operands) {
  IVLOG(5, "PrngOp::getResultType>")
  if (operands.size() < 1) {
    throw std::runtime_error("PrngOp requires at least one operand");
  }
  auto state = operands.front();
  auto dims = operands.drop_front();
  auto shape = eltwise::ComputeShape(dims);
  auto elementType = ScalarType::get(state->getContext(), DataType::f32);
  return RankedTensorType::get(shape, elementType);
}

//
// ---- ReshapeOp ----
//

struct ReshapeCanonicalizer : public OpRewritePattern<ReshapeOp> {
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(ReshapeOp reshapeOp, PatternRewriter& rewriter) const override {
    IVLOG(5, "ReshapeCanonicalizer::matchAndRewrite> " << mlir::debugString(reshapeOp));
    auto op = reshapeOp.getOperation();
    SmallVector<Value, 5> operands(op->getOperands());
    auto resultType = ReshapeOp::getResultType(operands);
    if (resultType == reshapeOp.result()->getType()) {
      return Pattern::matchFailure();
    }
    SmallVector<Value, 4> dims(reshapeOp.dims());
    auto newOp = rewriter.create<ReshapeOp>(op->getLoc(), resultType, reshapeOp.tensor(), dims);
    rewriter.replaceOp(op, {newOp});
    util::UpdateFuncOpType(newOp.getOperation());
    return Pattern::matchSuccess();
  }
};

void ReshapeOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<ReshapeCanonicalizer>(context);
}

Type ReshapeOp::getResultType(ArrayRef<Value> operands) {
  IVLOG(5, "ReshapeOp::getResultType>")
  if (operands.size() < 2) {
    throw std::runtime_error("ReshapeOp requires at least 2 operands");
  }
  auto tensor = operands.front();
  auto dims = operands.drop_front();
  auto tensorType = eltwise::getRankedTensorType(tensor->getType());
  auto elementType = tensorType.getElementType();
  auto shape = eltwise::ComputeShape(dims);
  return RankedTensorType::get(shape, elementType);
}

//
// --- ScatterOp ---
//

struct ScatterCanonicalizer : public OpRewritePattern<ScatterOp> {
  using OpRewritePattern<ScatterOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(ScatterOp scatterOp, PatternRewriter& rewriter) const override {
    IVLOG(5, "IndexCanonicalizer::matchAndRewrite> " << mlir::debugString(scatterOp));
    auto op = scatterOp.getOperation();
    SmallVector<Value, 3> operands(op->getOperands());
    auto resultType = ScatterOp::getResultType(operands);
    if (resultType == scatterOp.result()->getType()) {
      return Pattern::matchFailure();
    }
    auto newOp =
        rewriter.create<ScatterOp>(op->getLoc(), resultType, scatterOp.tensor(), scatterOp.dims(), scatterOp.other());
    rewriter.replaceOp(op, {newOp});
    util::UpdateFuncOpType(newOp.getOperation());
    return Pattern::matchSuccess();
  }
};

void ScatterOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<ScatterCanonicalizer>(context);
}

Type ScatterOp::getResultType(ArrayRef<Value> operands) {
  IVLOG(5, "ScatterOp::getResultType>")
  if (operands.size() != 3) {
    throw std::runtime_error("ScatterOp requires 3 operands");
  }
  auto tensor = operands[0];
  auto tensorType = eltwise::getRankedTensorType(tensor->getType());
  auto tensorElementType = tensorType.getElementType();
  const auto& tensorShape = tensorType.getShape();
  if (!tensorType.getRank()) {
    throw std::runtime_error("'scatter' requires first operand to have at least one dimension.");
  }
  auto index = operands[1];
  auto indexType = eltwise::getRankedTensorType(index->getType());
  auto indexElementType = indexType.getElementType().dyn_cast<ScalarType>();
  if (!indexElementType || indexElementType.type() != eltwise::DataType::i32) {
    throw std::runtime_error("'scatter' requires the data type for the second argument to be i32.");
  }
  auto other = operands[2];
  auto otherType = eltwise::getRankedTensorType(other->getType());
  const auto& otherShape = otherType.getShape();
  SmallVector<int64_t, 4> shape{otherShape[0]};
  for (unsigned i = indexType.getRank(); i < tensorType.getRank(); i++) {
    shape.emplace_back(tensorShape[i]);
  }
  auto resultType = RankedTensorType::get(shape, tensorElementType);
  IVLOG(6, "  resultType: " << mlir::debugString(resultType));
  return resultType;
}

//
// ---- ShapeOp ----
//

struct ShapeCanonicalizer : public OpRewritePattern<ShapeOp> {
  using OpRewritePattern<ShapeOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(ShapeOp shapeOp, PatternRewriter& rewriter) const override {
    IVLOG(5, "ShapeCanonicalizer::matchAndRewrite> " << mlir::debugString(shapeOp));
    auto op = shapeOp.getOperation();
    SmallVector<Value, 1> operands(op->getOperands());
    auto resultType = ShapeOp::getResultType(operands);
    if (resultType == shapeOp.result()->getType()) {
      return Pattern::matchFailure();
    }
    auto newOp = rewriter.create<ShapeOp>(op->getLoc(), resultType, shapeOp.tensor());
    rewriter.replaceOp(op, {newOp});
    util::UpdateFuncOpType(newOp.getOperation());
    return Pattern::matchSuccess();
  }
};

void ShapeOp::getCanonicalizationPatterns(OwningRewritePatternList& results, MLIRContext* context) {
  results.insert<ShapeCanonicalizer>(context);
}

Type ShapeOp::getResultType(ArrayRef<Value> operands) {
  IVLOG(5, "ShapeOp::getResultType>")
  if (operands.size() != 1) {
    throw std::runtime_error("ShapeOp requires 1 operand");
  }
  auto tensor = operands[0];
  auto tensorType = eltwise::getRankedTensorType(tensor->getType());
  auto elementType = ScalarType::get(tensor->getContext(), eltwise::DataType::i32);  // TODO: index type?
  return RankedTensorType::get({tensorType.getRank()}, elementType);
}

// ---- DimOp ----

IntegerAttr DimOp::resolve() {  //
  auto type = tensor()->getType().dyn_cast<mlir::TensorType>();
  if (!type) {
    return {};
  }
  auto size = type.getDimSize(dim().getSExtValue());
  if (mlir::ShapedType::isDynamic(size)) {
    return {};
  }
  auto indexType = mlir::IndexType::get(getContext());
  return IntegerAttr::get(indexType, size);
}

void printDimOp(OpAsmPrinter* printer, DimOp op) {  //
  *printer << op.getOperation()->getName() << ' ';
  printer->printOperand(op.tensor());
  *printer << '[' << op.dim().getZExtValue() << ']';
}

ParseResult parseDimOp(OpAsmParser* parser, OperationState& result) {
  auto indexType = parser->getBuilder().getIndexType();
  OpAsmParser::OperandType tensor;
  IntegerAttr dim;
  Type type;
  if (parser->parseOperand(tensor) ||                                      //
      parser->parseLSquare() ||                                            //
      parser->parseAttribute(dim, indexType, "dim", result.attributes) ||  //
      parser->parseRSquare() ||                                            //
      parser->parseColonType(type) ||                                      //
      parser->resolveOperand(tensor, type, result.operands)) {
    return failure();
  }
  result.addTypes(indexType);
  return success();
}

LogicalResult verifyDimOp(DimOp op) {  //
  return success();
}

// ---- AffineConstantOp ----

void printAffineConstantOp(OpAsmPrinter* printer, AffineConstantOp op) {  //
  *printer << op.getOperation()->getName() << ' ' << op.value().getZExtValue();
}

ParseResult parseAffineConstantOp(OpAsmParser* parser, OperationState& result) {
  auto indexType = parser->getBuilder().getIndexType();
  IntegerAttr value;
  if (parser->parseAttribute(value, indexType, "value", result.attributes)) {
    return failure();
  }
  result.addTypes(indexType);
  return success();
}

LogicalResult verifyAffineConstantOp(AffineConstantOp op) {  //
  return success();
}

// ---- AffineIndexOp ----

void printAffineIndexOp(OpAsmPrinter* printer, AffineIndexOp op) {  //
  *printer << op.getOperation()->getName() << ' ' << op.id().getSExtValue();
}

ParseResult parseAffineIndexOp(OpAsmParser* parser, OperationState& result) {
  IntegerAttr id;
  if (parser->parseAttribute(id, "id", result.attributes)) {
    return failure();
  }
  auto indexType = parser->getBuilder().getIndexType();
  result.addTypes(indexType);
  return success();
}

LogicalResult verifyAffineIndexOp(AffineIndexOp op) {  //
  return success();
}

// ---- AffineTensorMapOp ----

void printAffineTensorMapOp(OpAsmPrinter* printer, AffineTensorMapOp op) {
  *printer << op.getOperation()->getName() << ' ';
  printer->printOperand(op.tensor());
  *printer << '[';
  printer->printOperands(op.dims());
  *printer << ']';
}

ParseResult parseAffineTensorMapOp(OpAsmParser* parser, OperationState& result) {
  auto indexType = parser->getBuilder().getIndexType();
  OpAsmParser::OperandType tensor;
  SmallVector<OpAsmParser::OperandType, 4> dims;
  Type type;
  if (parser->parseOperand(tensor) ||                                    //
      parser->parseOperandList(dims, OpAsmParser::Delimiter::Square) ||  //
      parser->parseColonType(type) ||                                    //
      parser->resolveOperand(tensor, type, result.operands) ||           //
      parser->resolveOperands(dims, indexType, result.operands)) {
    return failure();
  }
  result.addTypes(AffineTensorMapType::get(result.getContext()));
  return success();
}

LogicalResult verifyAffineTensorMapOp(AffineTensorMapOp op) {  //
  return success();
}

// ---- AffineMapOp ----

void printAffineMapOp(OpAsmPrinter* printer, AffineMapOp op) {
  *printer << op.getOperation()->getName() << ' ';
  printer->printOperands(op.dims());
}

ParseResult parseAffineMapOp(OpAsmParser* parser, OperationState& result) {
  auto indexType = parser->getBuilder().getIndexType();
  SmallVector<OpAsmParser::OperandType, 4> dims;
  Type type;
  if (parser->parseOperandList(dims) ||  //
      parser->resolveOperands(dims, indexType, result.operands)) {
    return failure();
  }
  result.addTypes(AffineMapType::get(result.getContext()));
  return success();
}

LogicalResult verifyAffineMapOp(AffineMapOp op) {  //
  return success();
}

// ---- AffineConstraintsOp ----

void printAffineConstraintsOp(OpAsmPrinter* printer, AffineConstraintsOp op) {
  *printer << op.getOperation()->getName() << ' ';
  *printer << '(';
  printer->printOperands(op.pairs());
  *printer << ')';
}

ParseResult parseAffineConstraintsOp(OpAsmParser* parser, OperationState& result) {
  auto indexType = parser->getBuilder().getIndexType();
  SmallVector<OpAsmParser::OperandType, 4> dims;
  Type type;
  if (parser->parseOperandList(dims, OpAsmParser::Delimiter::Paren) ||  //
      parser->resolveOperands(dims, indexType, result.operands)) {
    return failure();
  }
  result.addTypes(AffineConstraintsType::get(result.getContext()));
  return success();
}

LogicalResult verifyAffineConstraintsOp(AffineConstraintsOp op) {  //
  return success();
}

// ---- SymbolicContractionOp ----

void printSymbolicContractionOp(OpAsmPrinter* printer, SymbolicContractionOp op) {  //
  *printer << op.getOperation()->getName() << ' ';
  *printer << util::stringifyAggregationKind(op.agg());
  *printer << ", ";
  *printer << util::stringifyCombinationKind(op.combo());
  *printer << ", ";
  printer->printOperand(op.init());
  *printer << ", ";
  printer->printOperand(op.cons());
  *printer << ", ";
  printer->printOperand(op.size());
  *printer << ", ";
  printer->printOperand(op.sink());
  *printer << ", ";
  printer->printOperands(op.srcs());
  *printer << " : ";
  printer->printType(op.init()->getType());
  *printer << " -> ";
  printer->printType(op.result()->getType());
}

ParseResult parseSymbolicContractionOp(OpAsmParser* parser, OperationState& result) {
  StringRef strAgg;
  StringRef strCombo;
  OpAsmParser::OperandType init;
  Type initType;
  OpAsmParser::OperandType cons;
  auto consType = AffineConstraintsType::get(result.getContext());
  OpAsmParser::OperandType size;
  auto mapType = AffineMapType::get(result.getContext());
  OpAsmParser::OperandType sink;
  auto tmapType = AffineTensorMapType::get(result.getContext());
  SmallVector<OpAsmParser::OperandType, 3> srcs;
  Type resultType;
  if (parser->parseKeyword(&strAgg) ||     //
      parser->parseComma() ||              //
      parser->parseKeyword(&strCombo) ||   //
      parser->parseComma() ||              //
      parser->parseOperand(init) ||        //
      parser->parseComma() ||              //
      parser->parseOperand(cons) ||        //
      parser->parseComma() ||              //
      parser->parseOperand(size) ||        //
      parser->parseComma() ||              //
      parser->parseOperand(sink) ||        //
      parser->parseComma() ||              //
      parser->parseOperandList(srcs) ||    //
      parser->parseColonType(initType) ||  //
      parser->parseArrow() ||              //
      parser->parseType(resultType)) {
    return failure();
  }

  if (parser->resolveOperand(init, initType, result.operands) ||
      parser->resolveOperand(cons, consType, result.operands) ||
      parser->resolveOperand(size, mapType, result.operands) ||
      parser->resolveOperand(sink, mapType, result.operands) ||
      parser->resolveOperands(srcs, tmapType, result.operands)) {
    return failure();
  }

  auto agg = util::symbolizeAggregationKind(strAgg);
  if (!agg) {
    failure();
  }
  result.addAttribute("agg", parser->getBuilder().getI64IntegerAttr(static_cast<int64_t>(agg.getValue())));

  auto combo = util::symbolizeCombinationKind(strCombo);
  if (!combo) {
    failure();
  }
  result.addAttribute("combo", parser->getBuilder().getI64IntegerAttr(static_cast<int64_t>(combo.getValue())));

  result.addTypes(resultType);
  return success();
}

LogicalResult verifySymbolicContractionOp(SymbolicContractionOp op) {  //
  return success();
}

// ---- ContractionOp ----

unsigned ContractionOp::getNumTensors() {  //
  return getNumTensors(combo());
}

unsigned ContractionOp::getNumSymbols() {  //
  return getNumOperands() - 1 - getNumTensors();
}

Value ContractionOp::getTensor(unsigned i) {  //
  return *std::next(operands().begin(), i);
}

Value ContractionOp::getSymbol(unsigned i) {  //
  return *std::next(operands().begin(), getNumTensors() + i);
}

void printContractionOp(OpAsmPrinter* printer, ContractionOp op) {
  SmallVector<StringRef, 3> elidedAttrs = {"agg", "combo", "name"};
  // if (op.cons().hasValue() && op.cons().getValue().isEmptyIntegerSet()) {
  //   elidedAttrs.emplace_back("cons");
  // }
  *printer << op.getOperation()->getName() << ' ';
  *printer << util::stringifyAggregationKind(op.agg());
  *printer << ", ";
  *printer << util::stringifyCombinationKind(op.combo());
  *printer << ", ";
  printer->printOperand(op.init());
  auto numTensors = op.getNumTensors();
  for (unsigned i = 0; i < numTensors; i++) {
    *printer << ", ";
    printer->printOperand(op.getTensor(i));
  }
  auto numSymbols = op.getNumSymbols();
  if (numSymbols) {
    *printer << " [";
    for (unsigned i = 0; i < numSymbols; i++) {
      if (i) {
        *printer << ", ";
      }
      printer->printOperand(op.getSymbol(i));
    }
    *printer << ']';
  }
  printer->printOptionalAttrDict(op.getAttrs(), elidedAttrs);
  *printer << " : ";
  printer->printType(op.init()->getType());
  *printer << ", ";
  for (unsigned i = 0; i < numTensors; i++) {
    if (i) {
      *printer << ", ";
    }
    printer->printType(op.getTensor(i)->getType());
  }
  *printer << " -> ";
  printer->printType(op.result()->getType());
}

ParseResult parseContractionOp(OpAsmParser* parser, OperationState& result) {
  StringRef strAgg;
  StringRef strCombo;
  OpAsmParser::OperandType init;
  SmallVector<OpAsmParser::OperandType, 3> tensors;
  SmallVector<OpAsmParser::OperandType, 8> symbols;
  SmallVector<Type, 4> types;
  Type resultType;
  if (parser->parseKeyword(&strAgg) ||    //
      parser->parseComma() ||             //
      parser->parseKeyword(&strCombo) ||  //
      parser->parseComma() ||             //
      parser->parseOperand(init) ||       //
      parser->parseComma()) {
    return failure();
  }

  auto agg = util::symbolizeAggregationKind(strAgg);
  if (!agg) {
    return failure();
  }
  result.addAttribute("agg", parser->getBuilder().getI64IntegerAttr(static_cast<int64_t>(agg.getValue())));

  auto combo = util::symbolizeCombinationKind(strCombo);
  if (!combo) {
    return failure();
  }
  result.addAttribute("combo", parser->getBuilder().getI64IntegerAttr(static_cast<int64_t>(combo.getValue())));

  auto numTensors = ContractionOp::getNumTensors(combo.getValue());
  if (parser->parseOperandList(tensors, numTensors) ||                              //
      parser->parseOperandList(symbols, OpAsmParser::Delimiter::OptionalSquare) ||  //
      parser->parseOptionalAttrDict(result.attributes) ||                           //
      parser->parseColonTypeList(types) ||                                          //
      parser->parseArrow() ||                                                       //
      parser->parseType(resultType)) {
    return failure();
  }

  auto loc = parser->getCurrentLocation();
  auto indexType = parser->getBuilder().getIndexType();
  auto tensorTypes = llvm::makeArrayRef(types).drop_front();
  if (parser->resolveOperand(init, types.front(), result.operands) ||
      parser->resolveOperands(tensors, tensorTypes, loc, result.operands) ||
      parser->resolveOperands(symbols, indexType, result.operands)) {
    return failure();
  }

  result.addTypes(resultType);
  return success();
}

bool isEltwiseAny(Type type) {
  if (auto rankedTensorType = type.dyn_cast<RankedTensorType>()) {
    auto elementType = rankedTensorType.getElementType();
    if (elementType.isa<IndexType>()) {
      return true;
    }
    if (auto scalarType = elementType.dyn_cast<ScalarType>()) {
      if (scalarType.type() != DataType::invalid) {
        return true;
      }
    }
  }
  return false;
}

LogicalResult verifyContractionOp(ContractionOp op) {
  auto numTensors = op.getNumTensors();
  auto numSymbols = op.getNumSymbols();
  SmallVector<Value, 8> variadic(op.operands());
  if (variadic.size() < numTensors) {
    return op.emitOpError("combo '") << util::stringifyCombinationKind(op.combo()) << "' requires " << numTensors
                                     << " tensor operands";
  }
  auto shape = op.shape();
  auto resultType = op.result()->getType().cast<RankedTensorType>();
  if (!resultType.hasStaticShape() && !shape.hasValue()) {
    return op.emitOpError("attribute 'shape' is required when result type is dynamic");
  }
  unsigned expectedSymbols = op.sink().getNumSymbols();
  if (shape.hasValue()) {
    expectedSymbols += shape->getNumSymbols();
  }
  for (auto src : op.srcs()) {
    auto map = src.cast<AffineMapAttr>();
    expectedSymbols += map.getValue().getNumSymbols();
  }
  if (op.cons().hasValue()) {
    expectedSymbols += op.cons().getValue().getNumSymbols();
  }
  if (expectedSymbols != numSymbols) {
    return op.emitOpError("has incorrect number of symbols: expected ")
           << expectedSymbols << " but found " << numSymbols;
  }
  for (unsigned i = 0; i < numTensors; i++) {
    auto type = op.getTensor(i)->getType();
    if (!isEltwiseAny(type)) {
      return op.emitOpError("tensor #") << i << " must be eltwise-any, but got " << type;
    }
  }
  for (unsigned i = 0; i < numSymbols; i++) {
    auto type = op.getSymbol(i)->getType();
    if (!type.isa<IndexType>()) {
      return op.emitOpError("symbol #") << i << " must be index, but got " << type;
    }
  }
  return success();
}

// ---- AffineBinaryOp ----

template <typename AffineBinaryOp>
void printAffineBinaryOp(OpAsmPrinter* printer, AffineBinaryOp op) {
  *printer << op.getOperation()->getName() << ' ';
  printer->printOperands(op.getOperands());
}

ParseResult parseAffineBinaryOp(OpAsmParser* parser, OperationState& result) {
  auto indexType = parser->getBuilder().getIndexType();
  OpAsmParser::OperandType lhs;
  OpAsmParser::OperandType rhs;
  if (parser->parseOperand(lhs) ||                                //
      parser->parseComma() ||                                     //
      parser->parseOperand(rhs) ||                                //
      parser->resolveOperand(lhs, indexType, result.operands) ||  //
      parser->resolveOperand(rhs, indexType, result.operands)) {
    return failure();
  }
  result.addTypes(IndexType::get(result.getContext()));
  return success();
}

#include "pmlc/dialect/tile/ir/interfaces.cc.inc"

#define GET_OP_CLASSES
#include "pmlc/dialect/tile/ir/ops.cc.inc"

}  // namespace pmlc::dialect::tile
