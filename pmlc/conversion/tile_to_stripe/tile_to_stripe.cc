// Copyright 2019, Intel Corporation

#include "pmlc/conversion/tile_to_stripe/tile_to_stripe.h"

#include <memory>
#include <unordered_set>
#include <vector>

#include "llvm/Support/FormatVariadic.h"

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Translation.h"

#include "base/util/logging.h"
#include "pmlc/dialect/eltwise/ir/dialect.h"
#include "pmlc/dialect/eltwise/ir/ops.h"
#include "pmlc/dialect/eltwise/ir/util.h"
#include "pmlc/dialect/stripe/affine_poly.h"
#include "pmlc/dialect/stripe/analysis.h"
#include "pmlc/dialect/stripe/dialect.h"
#include "pmlc/dialect/stripe/ops.h"
#include "pmlc/dialect/stripe/transcode.h"
#include "pmlc/dialect/stripe/util.h"
#include "pmlc/dialect/tile/ir/dialect.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/dialect/tile/transforms/contraction.h"
#include "pmlc/util/util.h"

using mlir::AbstractOperation;
using mlir::ArrayAttr;
using mlir::Block;
using mlir::ConversionPattern;
using mlir::ConversionPatternRewriter;
using mlir::ConversionTarget;
using mlir::FloatAttr;
using mlir::FuncOp;
using mlir::FunctionType;
using mlir::Identifier;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OwningModuleRef;
using mlir::PatternBenefit;
using mlir::PatternMatchResult;
using mlir::ReturnOp;
using mlir::UnknownLoc;
using mlir::Value;

namespace pmlc::dialect::tile {

static Shape getShape(Type type) {
  auto rankedTensorType = eltwise::getRankedTensorType(type);
  return rankedTensorType.getShape();
}

static stripe::TensorType convertIntoTensorType(Type type) {
  IVLOG(3, "convertIntoTensorType> " << mlir::debugString(type));
  auto rankedTensorType = eltwise::getRankedTensorType(type);
  auto shape = rankedTensorType.getShape();
  auto cls = Identifier::get(stripe::kAddressClassIdentifier, type.getContext());
  llvm::SmallVector<stripe::TensorDim, 4> newShape(shape.size(), stripe::TensorDim{0, 0, cls});
  // TODO: instead of using natural strides, use the I/O map supplied by the user
  int64_t stride = 1;
  for (int i = shape.size() - 1; i >= 0; i--) {
    newShape[i].stride = stride;
    newShape[i].size = shape[i];
    stride *= shape[i];
  }
  // TODO: deal with is_const
  return stripe::TensorType::get(rankedTensorType.getElementType(), newShape, stripe::OffsetsMap{}, false);
}

static eltwise::ScalarConstantOp createInit(OpBuilder* builder, Location loc, Type elementType, double value = 0.0) {
  auto scalarType = elementType.cast<ScalarType>();
  auto dtype = scalarType.type();
  if (is_float(dtype)) {
    auto constType = ScalarType::get(builder->getContext(), DataType::FLOAT32);
    return builder->create<eltwise::ScalarConstantOp>(loc, constType, value);
  }
  auto constType = ScalarType::get(builder->getContext(), DataType::INT32);
  return builder->create<eltwise::ScalarConstantOp>(loc, constType, static_cast<int64_t>(value));
}

struct TypeConverter : public mlir::TypeConverter {
  using mlir::TypeConverter::convertType;

  Type convertType(Type type) override {
    IVLOG(2, "TypeConverter::convertType> " << mlir::debugString(type));
    if (type.isa<FunctionType>()) {
      IVLOG(4, "  FunctionType");
      return type;
    }
    if (type.isa<RankedTensorType>()) {
      IVLOG(4, "  RankedTensorType");
      return stripe::TensorRefType::get(convertIntoTensorType(type));
    }
    if (type.isa<stripe::TensorType>()) {
      IVLOG(4, "  TensorType");
      return type;
    }
    if (type.isa<stripe::TensorRefType>()) {
      IVLOG(4, "  TensorRefType");
      return type;
    }
    return {};
  }
};

struct LoweringContext {
  explicit LoweringContext(MLIRContext* context) : context(context) {}

  MLIRContext* context;
  TypeConverter typeConverter;
  std::vector<StringAttr> names;
};

struct LoweringBase : public ConversionPattern {
  LoweringBase(                   //
      StringRef rootOpName,       //
      LoweringContext* lowering,  //
      PatternBenefit benefit = 1)
      : ConversionPattern(rootOpName, benefit, lowering->context), lowering(lowering) {}

  PatternMatchResult matchAndRewrite(  //
      Operation* op,                   //
      llvm::ArrayRef<Value> operands,  //
      ConversionPatternRewriter& rewriter) const final {
    try {
      return tryMatchAndRewrite(op, operands, rewriter);
    } catch (const std::exception& ex) {
      op->emitError(ex.what());
      return matchFailure();
    }
  }

 protected:
  virtual PatternMatchResult tryMatchAndRewrite(  //
      Operation* op,                              //
      llvm::ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const = 0;

 protected:
  LoweringContext* lowering;
};

struct AffineConstantOpConversion : public LoweringBase {
  explicit AffineConstantOpConversion(LoweringContext* lowering)  //
      : LoweringBase(AffineConstantOp::getOperationName(), lowering) {}

  PatternMatchResult tryMatchAndRewrite(  //
      Operation* op,                      //
      llvm::ArrayRef<Value> operands,     //
      ConversionPatternRewriter& rewriter) const override {
    IVLOG(2, "AffineConstantOpConversion::matchAndRewrite>");
    auto constOp = llvm::cast<AffineConstantOp>(op);
    auto poly = stripe::AffinePolynomial(constOp.value().getSExtValue());
    rewriter.replaceOpWithNewOp<stripe::AffinePolyOp>(op, poly);
    return matchSuccess();
  }
};

struct ReturnOpConversion : public LoweringBase {
  explicit ReturnOpConversion(LoweringContext* lowering)  //
      : LoweringBase(ReturnOp::getOperationName(), lowering) {}

  PatternMatchResult tryMatchAndRewrite(  //
      Operation* op,                      //
      llvm::ArrayRef<Value> operands,     //
      ConversionPatternRewriter& rewriter) const override {
    IVLOG(2, "ReturnOpConversion::matchAndRewrite>");
    rewriter.create<stripe::TerminateOp>(op->getLoc());
    rewriter.replaceOp(op, llvm::None);
    return matchSuccess();
  }
};

static Value findOutputArgument(Operation* op, Value tensor) {
  auto funcOp = op->getParentOfType<FuncOp>();
  auto attr = funcOp.getAttrOfType<IntegerAttr>("inputs");
  if (!attr) {
    throw std::runtime_error("Missing inputs attr");
  }
  for (auto user : tensor->getUsers()) {
    if (auto retOp = llvm::dyn_cast<ReturnOp>(user)) {
      for (unsigned i = 0; i < retOp.getNumOperands(); i++) {
        if (retOp.getOperand(i) == tensor) {
          return funcOp.getArgument(attr.getInt() + i);
        }
      }
    }
  }
  return nullptr;
}

static StringAttr getTensorName(LoweringContext* context, Value tensor) {
  if (auto op = tensor->getDefiningOp()) {
    if (op->getNumOperands()) {
      return getTensorName(context, op->getOperand(0));
    }
    return {};
  }
  auto blockArg = tensor.cast<mlir::BlockArgument>();
  return context->names[blockArg.getArgNumber()];
}

static Value MakeCombination(             //
    ConversionPatternRewriter* rewriter,  //
    Location loc,                         //
    CombinationKind combo,                //
    ScalarType scalarType,                //
    ArrayRef<Value> operands) {
  switch (combo) {
    case CombinationKind::none:
      return operands[0];
    case CombinationKind::add:
      return rewriter->create<eltwise::AddOp>(loc, scalarType, operands).result();
    case CombinationKind::cond: {
      auto constOp = createInit(rewriter, loc, scalarType);
      auto cmpEqOp = rewriter->create<eltwise::CmpEqOp>(loc, scalarType, operands.drop_back());
      llvm::SmallVector<Value, 3> args{cmpEqOp.result(), operands.back(), constOp.result()};
      return rewriter->create<eltwise::SelectOp>(loc, scalarType, args);
    }
    case CombinationKind::eq:
      return rewriter->create<eltwise::CmpEqOp>(loc, scalarType, operands).result();
    case CombinationKind::mul:
      return rewriter->create<eltwise::MulOp>(loc, scalarType, operands).result();
  }
  throw std::runtime_error("Invalid combination op");
}

struct ContractionOpConversion : public LoweringBase {
  explicit ContractionOpConversion(LoweringContext* lowering)
      : LoweringBase(ContractionOp::getOperationName(), lowering) {}

  PatternMatchResult tryMatchAndRewrite(  //
      Operation* op,                      //
      llvm::ArrayRef<Value> operands,     //
      ConversionPatternRewriter& rewriter) const override {
    IVLOG(2, "ContractionOpConversion::matchAndRewrite> " << mlir::debugString(*op));
    auto cionOp = llvm::cast<ContractionOp>(op);
    auto loc = op->getLoc();

    auto resultType = cionOp.result()->getType();
    auto resultTensorType = convertIntoTensorType(resultType);
    IVLOG(3, "resultTensorType: " << mlir::debugString(resultTensorType));
    llvm::SmallVector<Shape, 4> shapes{getShape(resultType)};
    llvm::SmallVector<std::vector<int64_t>, 4> tmpShapes;
    for (auto src : cionOp.operands()) {
      auto srcType = src->getType();
      if (srcType.isa<stripe::TensorRefType>()) {
        auto access = stripe::ComputeAccess(src);
        std::vector<int64_t> tmpShape;
        for (auto dim : access.base_type.getShape()) {
          tmpShape.push_back(dim.size);
        }
        tmpShapes.push_back(tmpShape);
        shapes.emplace_back(tmpShapes.back());
      } else {
        shapes.emplace_back(getShape(srcType));
      }
    }

    Contraction contraction{cionOp};
    bool no_reduce = cionOp.no_reduce().hasValue();
    const auto& [bounds, constraints] = contraction.ComputeBounds(shapes, no_reduce);

    // add induction vars
    llvm::SmallVector<int64_t, 8> ranges;
    for (const auto& [key, value] : bounds) {
      uint64_t range = value.max - value.min + 1;
      ranges.emplace_back(range);
    }

    auto outputTensor = findOutputArgument(op, cionOp.result());
    if (!outputTensor) {
      auto allocOp = rewriter.create<stripe::AllocateOp>(loc, resultTensorType);
      outputTensor = allocOp.result();
    }

    auto aggKind = cionOp.agg();
    auto aggOpTag = llvm::formatv("agg_op_{0}", util::stringifyAggregationKind(aggKind));
    auto comboKind = cionOp.combo();
    auto comboOpTag = llvm::formatv("comb_op_{0}", util::stringifyCombinationKind(comboKind));
    auto forOp = rewriter.create<stripe::ParallelForOp>(  //
        loc,                                              //
        rewriter.getI64ArrayAttr(ranges));
    auto attrs = llvm::SmallVector<NamedAttribute, 4>{
        {rewriter.getIdentifier("contraction"), rewriter.getUnitAttr()},
        {rewriter.getIdentifier("kernel"), rewriter.getUnitAttr()},
        {rewriter.getIdentifier(aggOpTag.str()), rewriter.getUnitAttr()},
        {rewriter.getIdentifier(comboOpTag.str()), rewriter.getUnitAttr()},
    };
    forOp.setAttr(stripe::Dialect::getStripeAttrsName(), rewriter.getDictionaryAttr(attrs));
    auto body = rewriter.createBlock(&forOp.inner());

    stripe::SymbolValueMap idxs;
    llvm::SmallVector<Attribute, 8> idxNames;
    for (const auto& [key, value] : bounds) {
      auto arg = body->addArgument(stripe::AffineType::get(rewriter.getContext()));
      idxNames.emplace_back(rewriter.getStringAttr(key));
      idxs.emplace(key, arg);
    }
    forOp.setAttr("idx_names", ArrayAttr::get(idxNames, rewriter.getContext()));

    llvm::SmallVector<Value, 4> tensors{outputTensor};
    for (auto src : cionOp.operands()) {
      tensors.emplace_back(src);
    }

    // add refinements
    Value output;
    llvm::SmallVector<Value, 4> refs;
    for (unsigned i = 0; i < contraction.accesses.size(); i++) {
      const auto& access = contraction.accesses[i];
      llvm::SmallVector<Value, 4> offsets;
      for (const auto& poly : access) {
        auto affine = Integerize(poly, bounds);
        offsets.emplace_back(stripe::AffineIntoMLIR(&rewriter, idxs, affine));
      }
      auto srcType = tensors[i]->getType();
      if (!srcType.isa<stripe::TensorRefType>()) {
        auto srcTensorType = convertIntoTensorType(srcType);
        srcType = stripe::TensorRefType::get(srcTensorType);
      }
      if (i) {
        auto tensorValue = tensors[i];
        auto tensorLoc = tensorValue->getLoc();
        auto tensorOp = tensorValue->getDefiningOp();
        if (tensorOp && llvm::isa<eltwise::ScalarConstantOp>(tensorOp)) {
          // This operand needs to be broadcast, skip creating a refinement
          refs.emplace_back(tensorValue);
        } else {
          auto refineOp = rewriter.create<stripe::RefineOp>(tensorLoc, srcType, tensorValue, offsets);
          auto refAttrs = llvm::SmallVector<NamedAttribute, 1>{
              {rewriter.getIdentifier("contraction"), rewriter.getUnitAttr()},
          };
          refineOp.setAttr(stripe::Dialect::getStripeAttrsName(), rewriter.getDictionaryAttr(refAttrs));
          if (auto name = getTensorName(lowering, tensorValue)) {
            refineOp.setAttr("name", name);
          }
          refs.emplace_back(refineOp.result());
        }
      } else {
        auto refineOp = rewriter.create<stripe::RefineOp>(loc, srcType, tensors[i], offsets);
        output = refineOp.result();
      }
    }

    // add constraints
    llvm::SmallVector<Value, 4> affineConstraints;
    for (const auto& constraint : constraints) {
      auto lhs = Integerize(constraint.poly, bounds);  // lhs <= rhs;
      lhs -= constraint.rhs;                           // lhs <= 0;
      lhs = -lhs;                                      // lhs >= 0
      IVLOG(3, "constraint: " << lhs << " >= 0");
      auto affine = stripe::AffineIntoMLIR(&rewriter, idxs, lhs);
      auto constraintOp = rewriter.create<stripe::ConstraintOp>(loc, affine);
      rewriter.create<stripe::TerminateOp>(loc);
      auto body = rewriter.createBlock(&constraintOp.ge_case());
      rewriter.setInsertionPointToStart(body);
      affineConstraints.emplace_back(affine);
    }

    if (needsInitialize(output, resultTensorType, ranges, affineConstraints)) {
      auto insertionPoint = rewriter.saveInsertionPoint();
      rewriter.setInsertionPoint(forOp.getOperation());
      addInitializer(&rewriter, cionOp, resultTensorType, outputTensor);
      rewriter.restoreInsertionPoint(insertionPoint);
      // TODO: ref_it->mut().set_tag("initialized");
    }

    // LOADs
    llvm::SmallVector<Value, 4> inputs;
    for (unsigned i = 0; i < refs.size(); i++) {
      auto defOp = refs[i]->getDefiningOp();
      if (defOp && llvm::isa<eltwise::ScalarConstantOp>(defOp)) {
        // No need to load a scalar constant, just use it directly
        // This happens when we want to broadcast a scalar constant value
        inputs.emplace_back(refs[i]);
      } else {
        auto srcType = refs[i]->getType();
        auto tensorRefType = srcType.cast<stripe::TensorRefType>();
        auto elementType = tensorRefType.getElementType();
        auto intoType = eltwise::getRankedTensorType(elementType);
        auto loadOp = rewriter.create<stripe::LoadOp>(loc, intoType, refs[i]);
        inputs.emplace_back(loadOp.into());
      }
    }

    // Combination Operation
    // TODO: verify correct scalarType
    auto scalarType = resultTensorType.getElementType().cast<eltwise::ScalarType>();
    auto combo = MakeCombination(&rewriter, cionOp.getLoc(), cionOp.combo(), scalarType, inputs);

    // STORE/Aggregate
    if (aggKind == AggregationKind::assign) {
      rewriter.create<stripe::StoreOp>(loc, output, combo);
    } else {
      auto aggAttr = rewriter.getI64IntegerAttr(static_cast<int>(aggKind));
      rewriter.create<stripe::AggregateOp>(loc, output, combo, aggAttr);
    }

    rewriter.create<stripe::TerminateOp>(loc);
    rewriter.replaceOp(op, {outputTensor});

    return matchSuccess();
  }

  bool needsInitialize(            //
      Value outRef,                //
      stripe::TensorType outType,  //
      ArrayRef<int64_t> ranges,    //
      ArrayRef<Value> constraints) const {
    // Check if we have a simple output: 1 unique index per dimension, each full range
    // If not, presume we need initialization for safety
    // We assume here that the 0'th refinement is the output refinement
    mlir::DenseSet<mlir::BlockArgument> outIdxs;
    auto refineOp = llvm::cast<stripe::RefineOp>(outRef->getDefiningOp());
    auto outShape = outType.getShape();
    for (unsigned i = 0; i < outShape.size(); i++) {
      stripe::AffinePolynomial poly{refineOp.getOffset(i)};
      if (poly == stripe::AffinePolynomial() && outShape[i].size == 1) {
        continue;
      }
      if (poly.constant || poly.terms.size() != 1 || poly.terms.begin()->second != 1) {
        // Not a single index with a multiplier of 1, bail
        return true;
      }
      // TODO: should we check all terms?
      auto idx = poly.terms.begin()->first;
      if (outIdxs.count(idx)) {
        // The index isn't unique, bail
        return true;
      }
      outIdxs.insert(idx);
      auto range = ranges[idx->getArgNumber()];
      if (range != outShape[i].size) {
        // The index range does not cover the entire size of the tensor dimension
        return true;
      }
    }

    // Now we check if we have any constraints that are 'output only'
    // Output only indexes actually reduce the range we write to, whereas constraints
    // that use both input + output make writes but only process some of the input
    for (auto constraint : constraints) {
      stripe::AffinePolynomial poly{constraint};
      bool any_inputs = false;
      for (const auto& [key, value] : poly.terms) {
        if (!outIdxs.count(key)) {
          any_inputs = true;
        }
      }
      if (!any_inputs) {
        // Found at least one output only constraint
        return true;
      }
    }
    // Looks good!
    return false;
  }

  void addInitializer(                //
      OpBuilder* builder,             //
      ContractionOp contractionOp,    //
      stripe::TensorType tensorType,  //
      Value output) const {
    auto loc = contractionOp.getLoc();
    stripe::SymbolValueMap idxs;
    llvm::SmallVector<int64_t, 6> ranges;
    llvm::SmallVector<Attribute, 6> idxNames;
    auto dims = tensorType.getShape();
    for (const auto& dim : dims) {
      ranges.emplace_back(dim.size);
    }

    auto forOp = builder->create<stripe::ParallelForOp>(loc, builder->getI64ArrayAttr(ranges));
    auto attrs = llvm::SmallVector<NamedAttribute, 2>{
        {builder->getIdentifier("kernel"), builder->getUnitAttr()},
    };
    auto body = builder->createBlock(&forOp.inner());
    builder->setInsertionPointToStart(body);

    llvm::SmallVector<Value, 6> offsets;
    for (unsigned i = 0; i < dims.size(); i++) {
      auto arg = body->addArgument(stripe::AffineType::get(builder->getContext()));
      auto idxName = llvm::formatv("i{0}", i);
      idxNames.emplace_back(builder->getStringAttr(idxName.str()));
      offsets.emplace_back(arg);
    }

    forOp.setAttr("idx_names", ArrayAttr::get(idxNames, builder->getContext()));

    auto refType = stripe::TensorRefType::get(tensorType);
    auto elementType = refType.getElementType();
    auto sink = builder->create<stripe::RefineOp>(loc, refType, output, offsets);
    auto init = contractionOp.init();
    auto defOp = init->getDefiningOp();

    IntegerAttr intAttr;
    FloatAttr floatAttr;
    if (defOp && mlir::m_Constant(&intAttr).match(defOp)) {
      attrs.emplace_back(builder->getIdentifier("zero"), builder->getUnitAttr());
      auto constOp = createInit(builder, loc, elementType, intAttr.getValue().getZExtValue());
      builder->create<stripe::StoreOp>(loc, sink.result(), constOp.result());
    } else if (defOp && mlir::m_Constant(&floatAttr).match(defOp)) {
      attrs.emplace_back(builder->getIdentifier("zero"), builder->getUnitAttr());
      auto constOp = createInit(builder, loc, elementType, floatAttr.getValue().convertToDouble());
      builder->create<stripe::StoreOp>(loc, sink.result(), constOp.result());
    } else {
      attrs.emplace_back(builder->getIdentifier("copy"), builder->getUnitAttr());
      auto src = builder->create<stripe::RefineOp>(loc, refType, init, offsets);
      auto intoType = eltwise::getRankedTensorType(elementType);
      auto loadOp = builder->create<stripe::LoadOp>(loc, intoType, src.result());
      builder->create<stripe::StoreOp>(loc, sink.result(), loadOp.into());
    }

    forOp.setAttr(stripe::Dialect::getStripeAttrsName(), builder->getDictionaryAttr(attrs));

    builder->create<stripe::TerminateOp>(loc);
  }
};

struct EltwiseOpConversion : public LoweringBase {
  explicit EltwiseOpConversion(LoweringContext* lowering, StringRef opName)  //
      : LoweringBase(opName, lowering) {}

  PatternMatchResult tryMatchAndRewrite(  //
      Operation* op,                      //
      llvm::ArrayRef<Value> operands,     //
      ConversionPatternRewriter& rewriter) const override {
    IVLOG(2, "EltwiseOpConversion::matchAndRewrite> " << mlir::debugString(*op));

    auto resultType = op->getResult(0)->getType();
    auto resultTensorType = convertIntoTensorType(resultType);
    IVLOG(3, "resultTensorType: " << mlir::debugString(resultTensorType));

    auto outputTensor = findOutputArgument(op, op->getResult(0));
    if (!outputTensor) {
      auto allocOp = rewriter.create<stripe::AllocateOp>(op->getLoc(), resultTensorType);
      outputTensor = allocOp.result();
    }

    llvm::SmallVector<int64_t, 8> ranges;
    for (const auto& dim : resultTensorType.getShape()) {
      ranges.emplace_back(dim.size);
    }

    auto forOp = rewriter.create<stripe::ParallelForOp>(  //
        op->getLoc(),                                     //
        rewriter.getI64ArrayAttr(ranges));
    auto eltwiseOpTag = llvm::formatv("eltwise_{0}", util::getOpName(op->getName()));
    auto attrs = llvm::SmallVector<NamedAttribute, 3>{
        {rewriter.getIdentifier("eltwise"), rewriter.getUnitAttr()},
        {rewriter.getIdentifier("kernel"), rewriter.getUnitAttr()},
        {rewriter.getIdentifier(eltwiseOpTag.str()), rewriter.getUnitAttr()},
    };
    forOp.setAttr(stripe::Dialect::getStripeAttrsName(), rewriter.getDictionaryAttr(attrs));
    auto body = rewriter.createBlock(&forOp.inner());

    stripe::SymbolValueMap idxs;
    llvm::SmallVector<Attribute, 8> idxNames;
    auto dims = resultTensorType.getShape();
    for (unsigned i = 0; i < dims.size(); i++) {
      auto arg = body->addArgument(stripe::AffineType::get(rewriter.getContext()));
      auto idxName = llvm::formatv("i{0}", i);
      idxNames.emplace_back(rewriter.getStringAttr(idxName.str()));
      idxs.emplace(idxName, arg);
    }
    forOp.setAttr("idx_names", ArrayAttr::get(idxNames, rewriter.getContext()));

    // output refinement
    Value output;
    {
      llvm::SmallVector<Value, 4> offsets;
      for (unsigned i = 0; i < resultTensorType.getRank(); i++) {
        offsets.emplace_back(body->getArgument(i));
      }
      Type refType = stripe::TensorRefType::get(resultTensorType);
      auto refineOp = rewriter.create<stripe::RefineOp>(op->getLoc(), refType, outputTensor, offsets);
      output = refineOp.result();
    }

    // input refinements
    auto inputs = llvm::SmallVector<Value, 4>();
    auto constType = ScalarType::get(rewriter.getContext(), DataType::INT32);
    for (auto operand : operands) {
      auto defOp = operand->getDefiningOp();
      if (defOp && llvm::isa<eltwise::ScalarConstantOp>(defOp)) {
        inputs.emplace_back(operand);
        continue;
      }

      Type operandType = operand->getType();
      stripe::TensorRefType tensorRefType;
      llvm::Optional<int64_t> constValue;
      if (auto affineType = operandType.dyn_cast<stripe::AffineType>()) {
        tensorRefType = stripe::TensorRefType::get(constType, 0, true);
        auto poly = stripe::AffinePolynomial(operand);
        constValue = poly.constant;
      } else {
        tensorRefType = operandType.dyn_cast<stripe::TensorRefType>();
        if (!tensorRefType) {
          throw std::runtime_error("EltwiseOpConversion expected TensorRefType for operand type");
        }
      }
      if (resultTensorType.getRank() < tensorRefType.getRank()) {
        throw std::runtime_error("Invalid eltwise op: result rank < operand rank");
      }
      llvm::SmallVector<Value, 4> offsets(tensorRefType.getRank());
      for (unsigned i = 0; i < tensorRefType.getRank(); i++) {
        // handle broadcasts
        unsigned j = resultTensorType.getRank() - i - 1;
        unsigned k = tensorRefType.getRank() - i - 1;
        offsets[k] = body->getArgument(j);
      }

      if (constValue) {
        auto constOp = rewriter.create<eltwise::ScalarConstantOp>(op->getLoc(), constType, *constValue);
        inputs.emplace_back(constOp);
      } else {
        auto refineOp = rewriter.create<stripe::RefineOp>(op->getLoc(), operandType, operand, offsets);
        auto refAttrs = llvm::SmallVector<NamedAttribute, 1>{
            {rewriter.getIdentifier(eltwiseOpTag.str()), rewriter.getUnitAttr()},
        };
        refineOp.setAttr(stripe::Dialect::getStripeAttrsName(), rewriter.getDictionaryAttr(refAttrs));
        auto elementType = tensorRefType.getElementType();
        auto intoType = eltwise::getRankedTensorType(elementType);
        // LOAD
        auto loadOp = rewriter.create<stripe::LoadOp>(op->getLoc(), intoType, refineOp.result());
        inputs.emplace_back(loadOp.into());
      }
    }

    // INTRINSIC
    auto scalarType = resultTensorType.getElementType().cast<eltwise::ScalarType>();
    auto abstractOp = op->getAbstractOperation();
    auto builder = abstractOp->getInterface<util::GenericBuilder>();
    if (!builder) {
      op->emitError("GenericBuilder expected for intrisnic");
      return matchFailure();
    }
    auto intrinsicOp = builder->create(&rewriter, op->getLoc(), scalarType, inputs);

    // STORE
    rewriter.create<stripe::StoreOp>(op->getLoc(), output, intrinsicOp->getResult(0));

    rewriter.create<stripe::TerminateOp>(op->getLoc());
    rewriter.replaceOp(op, {outputTensor});
    return matchSuccess();
  }
};

struct FuncOpConversion : public LoweringBase {
  explicit FuncOpConversion(LoweringContext* lowering)  //
      : LoweringBase(FuncOp::getOperationName(), lowering) {}

  PatternMatchResult tryMatchAndRewrite(  //
      Operation* op,                      //
      ArrayRef<Value> operands,           //
      ConversionPatternRewriter& rewriter) const override {
    auto funcOp = llvm::cast<FuncOp>(op);
    FunctionType type = funcOp.getType();
    IVLOG(2, "FuncOpConversion::matchAndRewrite> " << mlir::debugString(type));

    // Convert the original function arguments.
    std::vector<Type> tensorTypes;
    mlir::TypeConverter::SignatureConversion result(type.getNumInputs() + type.getNumResults());
    for (unsigned i = 0; i < type.getNumInputs(); ++i) {
      IVLOG(3, "  input(" << i << ")");
      auto tensorType = convertIntoTensorType(type.getInput(i));
      auto tensorRefType = stripe::TensorRefType::get(tensorType);
      result.addInputs(i, tensorRefType);
      tensorTypes.emplace_back(tensorType);
    }

    // append old results onto the end of new inputs
    for (unsigned i = 0; i < type.getNumResults(); i++) {
      IVLOG(3, "  result(" << i << ")");
      auto tensorType = convertIntoTensorType(type.getResult(i));
      auto tensorRefType = stripe::TensorRefType::get(tensorType);
      auto argIndex = type.getNumInputs() + i;
      result.addInputs(argIndex, tensorRefType);
      tensorTypes.emplace_back(tensorType);
    }

    // Create a new function with an updated signature.
    auto newFuncOp = rewriter.cloneWithoutRegions(funcOp);
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(), newFuncOp.end());
    newFuncOp.setType(FunctionType::get(result.getConvertedTypes(), llvm::None, funcOp.getContext()));
    auto attrs = llvm::SmallVector<NamedAttribute, 1>{
        {rewriter.getIdentifier("program"), rewriter.getUnitAttr()},
    };
    newFuncOp.setAttr(stripe::Dialect::getStripeAttrsName(), rewriter.getDictionaryAttr(attrs));
    newFuncOp.setAttr("inputs", rewriter.getI32IntegerAttr(type.getNumInputs()));
    newFuncOp.setAttr("outputs", rewriter.getI32IntegerAttr(type.getNumResults()));

    auto tileName = rewriter.getIdentifier(Dialect::getDialectAttrName("name"));
    auto stripeName = stripe::Dialect::getDialectAttrName("name");
    auto stripeLayout = stripe::Dialect::getDialectAttrName("layout");

    lowering->names.clear();
    for (unsigned i = 0; i < type.getNumInputs(); i++) {
      auto name = rewriter.getStringAttr(llvm::formatv("_X{0}", i).str());
      if (auto attr = funcOp.getArgAttrOfType<StringAttr>(i, tileName)) {
        name = attr;
        newFuncOp.removeArgAttr(i, tileName);
      }
      newFuncOp.setArgAttr(i, stripeName, name);
      lowering->names.push_back(name);
      newFuncOp.setArgAttr(i, stripeLayout, TypeAttr::get(tensorTypes[i]));
    }

    auto returnOp = llvm::cast<ReturnOp>(newFuncOp.getBody().front().getTerminator());
    for (unsigned i = 0; i < type.getNumResults(); i++) {
      auto argIndex = type.getNumInputs() + i;
      auto operand = returnOp.getOperand(i);
      auto name = rewriter.getStringAttr(llvm::formatv("_X{0}", argIndex).str());
      if (auto op = operand->getDefiningOp()) {
        if (auto attr = op->getAttrOfType<StringAttr>("name")) {
          name = attr;
        }
      }
      newFuncOp.setArgAttr(argIndex, stripeName, name);
      lowering->names.push_back(name);
      newFuncOp.setArgAttr(argIndex, stripeLayout, TypeAttr::get(tensorTypes[argIndex]));
    }

    // Tell the rewriter to convert the region signature.
    rewriter.applySignatureConversion(&newFuncOp.getBody(), result);

    // finally cause the old func op to be erased
    rewriter.replaceOp(op, llvm::None);

    return matchSuccess();
  }
};

struct IndexOpConversion : public LoweringBase {
  explicit IndexOpConversion(LoweringContext* lowering)  //
      : LoweringBase(IndexOp::getOperationName(), lowering) {}

  PatternMatchResult tryMatchAndRewrite(  //
      Operation* op,                      //
      llvm::ArrayRef<Value> operands,     //
      ConversionPatternRewriter& rewriter) const override {
    IVLOG(2, "IndexOpConversion::matchAndRewrite>");
    auto indexOp = llvm::cast<IndexOp>(op);

    auto resultType = indexOp.result()->getType();
    auto resultTensorType = convertIntoTensorType(resultType);
    IVLOG(3, "resultTensorType: " << mlir::debugString(resultTensorType));

    auto outputTensor = findOutputArgument(op, indexOp.result());
    if (!outputTensor) {
      auto allocOp = rewriter.create<stripe::AllocateOp>(op->getLoc(), resultTensorType);
      outputTensor = allocOp.result();
    }

    llvm::SmallVector<int64_t, 8> ranges;
    for (const auto& dim : resultTensorType.getShape()) {
      ranges.emplace_back(dim.size);
    }

    auto forOp = rewriter.create<stripe::ParallelForOp>(  //
        op->getLoc(),                                     //
        rewriter.getI64ArrayAttr(ranges));
    auto attrs = llvm::SmallVector<NamedAttribute, 3>{
        {rewriter.getIdentifier("eltwise"), rewriter.getUnitAttr()},
        {rewriter.getIdentifier("eltwise_index"), rewriter.getUnitAttr()},
        {rewriter.getIdentifier("kernel"), rewriter.getUnitAttr()},
    };
    forOp.setAttr(stripe::Dialect::getStripeAttrsName(), rewriter.getDictionaryAttr(attrs));
    auto body = rewriter.createBlock(&forOp.inner());

    stripe::SymbolValueMap idxs;
    llvm::SmallVector<Attribute, 8> idxNames;
    auto dims = resultTensorType.getShape();
    for (unsigned i = 0; i < dims.size(); i++) {
      auto arg = body->addArgument(stripe::AffineType::get(rewriter.getContext()));
      auto idxName = llvm::formatv("i{0}", i);
      idxNames.emplace_back(rewriter.getStringAttr(idxName.str()));
      idxs.emplace(idxName, arg);
    }
    forOp.setAttr("idx_names", ArrayAttr::get(idxNames, rewriter.getContext()));

    // output refinement
    Value output;
    llvm::SmallVector<Value, 4> offsets;
    for (unsigned i = 0; i < resultTensorType.getRank(); i++) {
      offsets.emplace_back(body->getArgument(i));
    }
    Type refType = stripe::TensorRefType::get(resultTensorType);
    auto refineOp = rewriter.create<stripe::RefineOp>(op->getLoc(), refType, outputTensor, offsets);
    output = refineOp.result();

    // LOAD_INDEX
    auto elementType = resultTensorType.getElementType();
    auto loadType = eltwise::getRankedTensorType(elementType);
    auto dim = indexOp.dim().getZExtValue();
    auto loadIndexOp = rewriter.create<stripe::LoadIndexOp>(op->getLoc(), loadType, offsets[dim]);

    // STORE
    rewriter.create<stripe::StoreOp>(op->getLoc(), output, loadIndexOp.into());

    rewriter.create<stripe::TerminateOp>(op->getLoc());
    rewriter.replaceOp(op, {outputTensor});
    return matchSuccess();
  }
};

struct SpecialOpConversion : public LoweringBase {
  explicit SpecialOpConversion(LoweringContext* lowering, StringRef opName)  //
      : LoweringBase(opName, lowering) {}

  PatternMatchResult tryMatchAndRewrite(  //
      Operation* op,                      //
      ArrayRef<Value> operands,           //
      ConversionPatternRewriter& rewriter) const override {
    IVLOG(2, "SpecialOpConversion::matchAndRewrite>");

    auto opName = stripe::Dialect::getCanonicalOpName(util::getOpName(op->getName()));
    auto abstractOp = AbstractOperation::lookup(opName, op->getContext());
    if (!abstractOp) {
      op->emitError("AbstractOperation::lookup failed for: ") << opName;
      return matchFailure();
    }
    auto specialOp = abstractOp->getInterface<stripe::SpecialOp>();
    if (!specialOp) {
      op->emitError("SpecialOp interface expected");
      return matchFailure();
    }

    std::vector<Value> inputs;
    for (unsigned i = 0; i < specialOp->getNumInputs(); i++) {
      inputs.emplace_back(operands[i]);
    }

    std::vector<Value> outputs;
    for (unsigned i = 0; i < specialOp->getNumOutputs(); i++) {
      auto result = op->getResult(i);
      auto output = findOutputArgument(op, result);
      if (!output) {
        auto resultType = result->getType();
        auto resultTensorType = convertIntoTensorType(resultType);
        auto allocOp = rewriter.create<stripe::AllocateOp>(op->getLoc(), resultTensorType);
        output = allocOp.result();
      }
      outputs.emplace_back(output);
    }

    specialOp->create(&rewriter, op->getLoc(), inputs, outputs);
    rewriter.replaceOp(op, outputs);
    return matchSuccess();
  }
};

struct PrngOpConversion : public LoweringBase {
  explicit PrngOpConversion(LoweringContext* lowering, StringRef opName)  //
      : LoweringBase(opName, lowering) {}

  PatternMatchResult tryMatchAndRewrite(  //
      Operation* op,                      //
      ArrayRef<Value> operands,           //
      ConversionPatternRewriter& rewriter) const override {
    IVLOG(2, "PrngOpConversion::matchAndRewrite>");
    auto prngOp = llvm::cast<PrngOp>(op);

    auto opName = stripe::Dialect::getCanonicalOpName("prng_step");
    auto abstractOp = AbstractOperation::lookup(opName, op->getContext());
    if (!abstractOp) {
      op->emitError("AbstractOperation::lookup failed for: ") << opName;
      return matchFailure();
    }

    auto specialOp = abstractOp->getInterface<stripe::SpecialOp>();
    if (!specialOp) {
      op->emitError("SpecialOp interface expected");
      return matchFailure();
    }

    std::vector<Value> inputs{prngOp.state()};
    std::vector<Value> outputs;
    auto newState = findOutputArgument(op, prngOp.new_state());
    outputs.emplace_back(newState);
    auto result = prngOp.result();
    auto output = findOutputArgument(op, result);
    if (!output) {
      auto resultType = result->getType();
      auto resultTensorType = convertIntoTensorType(resultType);
      auto allocOp = rewriter.create<stripe::AllocateOp>(op->getLoc(), resultTensorType);
      output = allocOp.result();
    }
    outputs.emplace_back(output);

    specialOp->create(&rewriter, op->getLoc(), inputs, outputs);
    result->replaceAllUsesWith(output);
    prngOp.new_state()->replaceAllUsesWith(newState);
    rewriter.eraseOp(op);
    return matchSuccess();
  }
};

struct LoweringPass : public mlir::ModulePass<LoweringPass> {
  void runOnModule() override {
    LoweringContext lowering{&getContext()};
    ConversionTarget target(getContext());
    target.addLegalDialect<stripe::Dialect>();
    std::function<bool(Operation*)> isDynamicallyLegal = [](Operation* op) {
      IVLOG(3, "isDynamicallyLegal: " << op->getName().getStringRef().str());
      return llvm::isa<eltwise::EltwiseOp>(op) && op->getParentOfType<stripe::ParallelForOp>();
    };
    target.addLegalOp<eltwise::ScalarConstantOp>();
    target.addDynamicallyLegalDialect<eltwise::Dialect>(isDynamicallyLegal);
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      auto funcType = op.getType();
      if (funcType.getNumResults()) {
        return false;
      }
      return lowering.typeConverter.isSignatureLegal(funcType);
    });

    OwningRewritePatternList patterns;
    patterns.insert<                 //
        AffineConstantOpConversion,  //
        ContractionOpConversion,     //
        FuncOpConversion,            //
        IndexOpConversion,           //
        ReturnOpConversion>(&lowering);
    auto eltwiseOps = util::getAllOpsWithInterface<eltwise::EltwiseOp>(&getContext());
    for (auto op : eltwiseOps) {
      patterns.insert<EltwiseOpConversion>(&lowering, op->name);
    }
    auto specialOps = util::getAllOpsWithInterface<tile::SpecialOp>(&getContext());
    for (auto op : specialOps) {
      if (op->name == tile::Dialect::getCanonicalOpName("prng")) {
        patterns.insert<PrngOpConversion>(&lowering, op->name);
      } else {
        patterns.insert<SpecialOpConversion>(&lowering, op->name);
      }
    }
    if (failed(applyPartialConversion(getModule(), target, patterns, &lowering.typeConverter))) {
      emitError(UnknownLoc::get(&getContext()), "Error lowering tile -> stripe\n");
      signalPassFailure();
    }

    getModule().walk([&](FuncOp funcOp) {
      // Wraps function body with a ParallelForOp to represent Stripe's 'main' block.
      stripe::createMainParallelFor(funcOp);
    });
  }

  static std::unique_ptr<mlir::Pass> Create() { return std::make_unique<LoweringPass>(); }
};

OwningModuleRef LowerIntoStripe(ModuleOp workspace) {
  IVLOG(1, "LowerIntoStripe");
  OwningModuleRef module(llvm::cast<ModuleOp>(workspace.getOperation()->clone()));
  mlir::PassManager pm(workspace.getContext());
  IVLOG(3, "before:\n" << mlir::debugString(*module->getOperation()));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(LoweringPass::Create());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  auto result = pm.run(*module);
  if (failed(result)) {
    IVLOG(1, "LowerIntoStripe failed: " << mlir::debugString(*module->getOperation()));
    throw std::runtime_error("Lowering to stripe dialect failure");
  }
  IVLOG(3, "after:\n" << mlir::debugString(*module->getOperation()));
  return module;
}

static mlir::LogicalResult IntoStripeTranslateFunction(mlir::ModuleOp input, llvm::raw_ostream& output) {
  auto module = LowerIntoStripe(input);
  auto stripe = stripe::FromMLIR(*module);
  std::stringstream ss;
  ss << *stripe->entry;
  output << ss.str();
  return mlir::success();
}

static mlir::PassRegistration<LoweringPass> legalize_pass(  //
    "tile-legalize-to-stripe",                              //
    "Legalize from Tile dialect to Stripe dialect");

static mlir::TranslateFromMLIRRegistration IntoStripeTranslate("tile-to-stripe", IntoStripeTranslateFunction);

}  // namespace pmlc::dialect::tile
