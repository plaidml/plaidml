// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/lowering.h"

#include <memory>
#include <vector>

#include "llvm/Support/FormatVariadic.h"

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Translation.h"

#include "base/util/logging.h"
#include "pmlc/dialect/eltwise/dialect.h"
#include "pmlc/dialect/eltwise/ops.h"
#include "pmlc/dialect/eltwise/util.h"
#include "pmlc/dialect/stripe/dialect.h"
#include "pmlc/dialect/stripe/ops.h"
#include "pmlc/dialect/stripe/transcode.h"
#include "pmlc/dialect/stripe/util.h"
#include "pmlc/dialect/tile/contraction.h"
#include "pmlc/dialect/tile/ops.h"
#include "pmlc/dialect/tile/program.h"
#include "pmlc/util/util.h"

using mlir::AbstractOperation;
using mlir::ArrayAttr;
using mlir::Block;
using mlir::ConversionPattern;
using mlir::ConversionPatternRewriter;
using mlir::ConversionTarget;
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

namespace pmlc {
namespace dialect {
namespace tile {

static stripe::TensorType convertIntoTensorType(Type type) {
  if (auto rankedType = type.dyn_cast<RankedTensorType>()) {
    auto shape = rankedType.getShape();
    auto cls = Identifier::get(stripe::kAddressClassIdentifier, rankedType.getContext());
    llvm::SmallVector<stripe::TensorDim, 4> newShape(shape.size(), stripe::TensorDim{0, 0, cls});
    // TODO: instead of using natural strides, use the I/O map supplied by the user
    int64_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; i--) {
      newShape[i].stride = stride;
      newShape[i].size = shape[i];
      stride *= shape[i];
    }
    // TODO: deal with is_const
    return stripe::TensorType::get(rankedType.getElementType(), newShape, stripe::OffsetsMap{}, false);
  }
  throw std::runtime_error("Invalid type");
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
};

struct LoweringBase : public ConversionPattern {
  LoweringBase(                   //
      StringRef rootOpName,       //
      LoweringContext* lowering,  //
      PatternBenefit benefit = 1)
      : ConversionPattern(rootOpName, benefit, lowering->context), lowering(lowering) {}

  PatternMatchResult matchAndRewrite(   //
      Operation* op,                    //
      llvm::ArrayRef<Value*> operands,  //
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
      llvm::ArrayRef<Value*> operands,
      ConversionPatternRewriter& rewriter) const = 0;  // NOLINT

 protected:
  LoweringContext* lowering;
};

struct AffineConstantOpConversion : public LoweringBase {
  explicit AffineConstantOpConversion(LoweringContext* lowering)  //
      : LoweringBase(AffineConstantOp::getOperationName(), lowering) {}

  PatternMatchResult tryMatchAndRewrite(  //
      Operation* op,                      //
      llvm::ArrayRef<Value*> operands,    //
      ConversionPatternRewriter& rewriter) const override {
    IVLOG(2, "AffineConstantOpConversion::matchAndRewrite>");
    auto constOp = llvm::cast<AffineConstantOp>(op);
    auto newOp = rewriter.create<stripe::AffineConstOp>(  //
        op->getLoc(),                                     //
        rewriter.getType<stripe::AffineType>(),           //
        rewriter.getI64IntegerAttr(constOp.value().getSExtValue()));
    rewriter.replaceOp(op, {newOp});
    return matchSuccess();
  }
};

struct ReturnOpConversion : public LoweringBase {
  explicit ReturnOpConversion(LoweringContext* lowering)  //
      : LoweringBase(ReturnOp::getOperationName(), lowering) {}

  PatternMatchResult tryMatchAndRewrite(  //
      Operation* op,                      //
      llvm::ArrayRef<Value*> operands,    //
      ConversionPatternRewriter& rewriter) const override {
    IVLOG(2, "ReturnOpConversion::matchAndRewrite>");
    rewriter.create<stripe::TerminateOp>(op->getLoc());
    rewriter.replaceOp(op, llvm::None);
    return matchSuccess();
  }
};

static Value* ConvertOutputTensor(Operation* op, Value* tensor) {
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

static Value* MakeCombination(            //
    ConversionPatternRewriter* rewriter,  //
    ContractionOp op,                     //
    ScalarType scalarType,                //
    ArrayRef<Value*> operands) {
  switch (op.getCombinationKind()) {
    case CombinationKind::none:
      return operands[0];
    case CombinationKind::add:
      return rewriter->create<eltwise::AddOp>(op.getLoc(), scalarType, operands).result();
    case CombinationKind::cond:
      break;
    case CombinationKind::eq:
      return rewriter->create<eltwise::CmpEqOp>(op.getLoc(), scalarType, operands).result();
    case CombinationKind::mul:
      return rewriter->create<eltwise::MulOp>(op.getLoc(), scalarType, operands).result();
  }
  throw std::runtime_error("Invalid combination op");
}

struct AffineDomainOpConversion : public LoweringBase {
  explicit AffineDomainOpConversion(LoweringContext* lowering)  //
      : LoweringBase(AffineDomainOp::getOperationName(), lowering) {}

  PatternMatchResult tryMatchAndRewrite(  //
      Operation* op,                      //
      llvm::ArrayRef<Value*> operands,    //
      ConversionPatternRewriter& rewriter) const override {
    IVLOG(2, "AffineDomainOpConversion::matchAndRewrite>");
    auto domainOp = llvm::cast<AffineDomainOp>(op);

    std::vector<ConstraintOp> constraintOps;
    auto terminator = domainOp.body().front().getTerminator();
    while (auto constraintOp = llvm::dyn_cast<ConstraintOp>(terminator)) {
      terminator = constraintOp.body().front().getTerminator();
      constraintOps.emplace_back(constraintOp);
    }

    auto contractionOp = llvm::dyn_cast<ContractionOp>(terminator);
    if (!contractionOp) {
      return matchFailure();
    }

    auto resultType = domainOp.result()->getType();
    auto resultTensorType = convertIntoTensorType(resultType);
    IVLOG(3, "resultTensorType: " << mlir::debugString(resultTensorType));
    llvm::SmallVector<stripe::TensorType, 4> shapes{resultTensorType};
    for (auto src : contractionOp.getSourceIndexMaps()) {
      auto srcOp = llvm::cast<AffineSourceIndexMapOp>(src->getDefiningOp());
      auto srcType = srcOp.tensor()->getType();
      auto srcTensorType = convertIntoTensorType(srcType);
      shapes.push_back(srcTensorType);
    }
    Contraction contraction{contractionOp, constraintOps};
    IndexBounds bounds;
    SimpleConstraints constraints;
    std::tie(bounds, constraints) = contraction.ComputeBounds(shapes);

    // add induction vars
    llvm::SmallVector<int64_t, 8> ranges;
    for (const auto& kvp : bounds) {
      uint64_t range = kvp.second.max - kvp.second.min + 1;
      ranges.emplace_back(range);
    }

    auto outputTensor = ConvertOutputTensor(op, domainOp.result());
    if (!outputTensor) {
      auto allocOp = rewriter.create<stripe::AllocateOp>(op->getLoc(), resultTensorType);
      outputTensor = allocOp.result();
    }

    auto aggKind = contractionOp.getAggregationKind();
    auto aggOpTag = llvm::formatv("agg_op_{0}", util::stringifyAggregationKind(aggKind));
    auto comboKind = contractionOp.getCombinationKind();
    auto comboOpTag = llvm::formatv("combo_op_{0}", util::stringifyCombinationKind(comboKind));
    auto forOp = rewriter.create<stripe::ParallelForOp>(  //
        op->getLoc(),                                     //
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
    for (const auto& kvp : bounds) {
      auto arg = body->addArgument(stripe::AffineType::get(rewriter.getContext()));
      idxNames.emplace_back(rewriter.getStringAttr(kvp.first));
      idxs.emplace(kvp.first, arg);
    }
    forOp.setAttr("idx_names", ArrayAttr::get(idxNames, rewriter.getContext()));

    llvm::SmallVector<Value*, 4> tensors{outputTensor};
    for (auto src : contractionOp.getSourceIndexMaps()) {
      auto srcOp = llvm::cast<AffineSourceIndexMapOp>(src->getDefiningOp());
      tensors.emplace_back(srcOp.tensor());
    }

    // add refinements
    Value* output = nullptr;
    llvm::SmallVector<Value*, 4> refs;
    for (unsigned i = 0; i < contraction.accesses.size(); i++) {
      const auto& access = contraction.accesses[i];
      llvm::SmallVector<Value*, 4> offsets;
      for (const auto& poly : access) {
        auto affine = Integerize(poly, bounds);
        offsets.emplace_back(stripe::AffineIntoMLIR(&rewriter, forOp.getOperation(), idxs, affine));
      }
      auto srcType = tensors[i]->getType();
      if (!srcType.isa<stripe::TensorRefType>()) {
        auto srcTensorType = convertIntoTensorType(srcType);
        srcType = stripe::TensorRefType::get(srcTensorType);
      }
      auto refineOp = rewriter.create<stripe::RefineOp>(op->getLoc(), srcType, tensors[i], offsets);
      if (i) {
        auto refAttrs = llvm::SmallVector<NamedAttribute, 1>{
            {rewriter.getIdentifier("contraction"), rewriter.getUnitAttr()},
        };
        refineOp.setAttr(stripe::Dialect::getStripeAttrsName(), rewriter.getDictionaryAttr(refAttrs));
        refs.emplace_back(refineOp.result());
      } else {
        output = refineOp.result();
      }
    }

    // add constraints
    for (const auto& constraint : constraints) {
      auto lhs = Integerize(constraint.poly, bounds);  // lhs <= rhs;
      lhs -= constraint.rhs;                           // lhs <= 0;
      lhs = -lhs;                                      // lhs >= 0
      IVLOG(3, "constraint: " << lhs << " >= 0");
      auto affine = stripe::AffineIntoMLIR(&rewriter, forOp.getOperation(), idxs, lhs);
      auto constraintOp = rewriter.create<stripe::ConstraintOp>(op->getLoc(), affine);
      rewriter.create<stripe::TerminateOp>(op->getLoc());
      auto body = rewriter.createBlock(&constraintOp.ge_case());
      rewriter.setInsertionPointToStart(body);
    }

    // LOADs
    llvm::SmallVector<Value*, 4> inputs;
    for (unsigned i = 0; i < refs.size(); i++) {
      auto srcType = tensors[i]->getType();
      if (!srcType.isa<stripe::TensorRefType>()) {
        auto srcTensorType = convertIntoTensorType(srcType);
        srcType = stripe::TensorRefType::get(srcTensorType);
      }
      auto tensorRefType = srcType.cast<stripe::TensorRefType>();
      auto elementType = tensorRefType.getElementType();
      auto intoType = eltwise::getRankedTensorType(elementType);
      auto loadOp = rewriter.create<stripe::LoadOp>(op->getLoc(), intoType, refs[i]);
      inputs.emplace_back(loadOp.into());
    }

    // Combination Operation
    auto scalarType = resultTensorType.getElementType().cast<eltwise::ScalarType>();
    auto combo = MakeCombination(&rewriter, contractionOp, scalarType, inputs);

    // STORE/Aggregate
    if (aggKind == AggregationKind::assign) {
      rewriter.create<stripe::StoreOp>(op->getLoc(), output, combo);
    } else {
      auto aggAttr = rewriter.getI64IntegerAttr(static_cast<int>(aggKind));
      rewriter.create<stripe::AggregateOp>(op->getLoc(), output, combo, aggAttr);
    }

    rewriter.create<stripe::TerminateOp>(op->getLoc());
    rewriter.replaceOp(op, {outputTensor});

    // TODO: use_default
    // TODO: no_defract
    // TODO: NeedsInitialize

    return matchSuccess();
  }
};

struct EltwiseOpConversion : public LoweringBase {
  explicit EltwiseOpConversion(LoweringContext* lowering, StringRef opName)  //
      : LoweringBase(opName, lowering) {}

  PatternMatchResult tryMatchAndRewrite(  //
      Operation* op,                      //
      llvm::ArrayRef<Value*> operands,    //
      ConversionPatternRewriter& rewriter) const override {
    IVLOG(2, "EltwiseOpConversion::matchAndRewrite> " << mlir::debugString(*op));

    auto resultType = op->getResult(0)->getType();
    auto resultTensorType = convertIntoTensorType(resultType);
    IVLOG(3, "resultTensorType: " << mlir::debugString(resultTensorType));

    auto outputTensor = ConvertOutputTensor(op, op->getResult(0));
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
    const auto& dims = resultTensorType.getShape();
    for (unsigned i = 0; i < dims.size(); i++) {
      auto arg = body->addArgument(stripe::AffineType::get(rewriter.getContext()));
      auto idxName = llvm::formatv("i{0}", i);
      idxNames.emplace_back(rewriter.getStringAttr(idxName.str()));
      idxs.emplace(idxName, arg);
    }
    forOp.setAttr("idx_names", ArrayAttr::get(idxNames, rewriter.getContext()));

    // output refinement
    Value* output;
    {
      llvm::SmallVector<Value*, 4> offsets;
      for (unsigned i = 0; i < resultTensorType.getRank(); i++) {
        offsets.emplace_back(body->getArgument(i));
      }
      Type refType = stripe::TensorRefType::get(resultTensorType);
      auto refineOp = rewriter.create<stripe::RefineOp>(op->getLoc(), refType, outputTensor, offsets);
      output = refineOp.result();
    }

    // input refinements
    auto inputs = llvm::SmallVector<Value*, 4>();
    for (auto operand : operands) {
      auto defOp = operand->getDefiningOp();
      if (defOp && llvm::isa<eltwise::ScalarConstantOp>(defOp)) {
        inputs.emplace_back(operand);
        continue;
      }

      Type operandType = operand->getType();
      auto tensorRefType = operandType.cast<stripe::TensorRefType>();
      if (resultTensorType.getRank() < tensorRefType.getRank()) {
        throw std::runtime_error("Invalid eltwise op: result rank < operand rank");
      }
      llvm::SmallVector<Value*, 4> offsets(tensorRefType.getRank());
      for (unsigned i = 0; i < tensorRefType.getRank(); i++) {
        // handle broadcasts
        unsigned j = resultTensorType.getRank() - i - 1;
        unsigned k = tensorRefType.getRank() - i - 1;
        offsets[k] = body->getArgument(j);
      }

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
      ArrayRef<Value*> operands,          //
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

    for (unsigned i = 0; i < tensorTypes.size(); i++) {
      auto name = llvm::formatv("_X{0}", i);
      auto attrName = stripe::Dialect::getDialectAttrName("name");
      newFuncOp.setArgAttr(i, attrName, rewriter.getStringAttr(name.str()));
      auto attrLayout = stripe::Dialect::getDialectAttrName("layout");
      newFuncOp.setArgAttr(i, attrLayout, rewriter.getTypeAttr(tensorTypes[i]));
    }

    // Tell the rewriter to convert the region signature.
    rewriter.applySignatureConversion(&newFuncOp.getBody(), result);

    // finally cause the old func op to be erased
    rewriter.replaceOp(op, llvm::None);

    return matchSuccess();
  }
};

struct SpecialOpConversion : public LoweringBase {
  explicit SpecialOpConversion(LoweringContext* lowering, StringRef opName)  //
      : LoweringBase(opName, lowering) {}

  PatternMatchResult tryMatchAndRewrite(  //
      Operation* op,                      //
      ArrayRef<Value*> operands,          //
      ConversionPatternRewriter& rewriter) const override {
    IVLOG(2, "SpecialOpConversion::matchAndRewrite>");

    auto opName = stripe::Dialect::getCanonicalOpName(util::getOpName(op->getName()));
    auto abstractOp = AbstractOperation::lookup(opName, op->getContext());
    if (!abstractOp) {
      op->emitError("AbstractOperation::lookup failed for: " + opName);
      return matchFailure();
    }
    auto specialOp = abstractOp->getInterface<stripe::SpecialOp>();
    if (!specialOp) {
      op->emitError("SpecialOp interface expected");
      return matchFailure();
    }

    std::vector<Value*> inputs;
    for (unsigned i = 0; i < specialOp->getNumInputs(); i++) {
      inputs.emplace_back(operands[i]);
    }

    std::vector<Value*> outputs;
    for (unsigned i = 0; i < specialOp->getNumOutputs(); i++) {
      auto result = op->getResult(i);
      auto resultType = result->getType();
      auto resultTensorType = convertIntoTensorType(resultType);
      auto output = ConvertOutputTensor(op, result);
      if (!output) {
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

struct LoweringPass : public mlir::ModulePass<LoweringPass> {
  void runOnModule() override {
    LoweringContext lowering{&getContext()};
    ConversionTarget target(getContext());
    target.addLegalDialect<stripe::Dialect>();
    std::function<bool(Operation*)> isDynamicallyLegal = [](Operation* op) {
      IVLOG(3, "isDynamicallyLegal: " << op->getName().getStringRef().str());
      return llvm::isa<eltwise::EltwiseOp>(op) &&
             (op->getParentOfType<stripe::ParallelForOp>() || op->getParentOfType<stripe::ConstraintOp>());
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
        AffineDomainOpConversion,    //
        FuncOpConversion,            //
        ReturnOpConversion>(&lowering);
    auto eltwiseOps = util::getAllOpsWithInterface<eltwise::EltwiseOp>(&getContext());
    for (auto op : eltwiseOps) {
      patterns.insert<EltwiseOpConversion>(&lowering, op->name);
    }
    auto specialOps = util::getAllOpsWithInterface<tile::SpecialOp>(&getContext());
    for (auto op : specialOps) {
      patterns.insert<SpecialOpConversion>(&lowering, op->name);
    }
    if (failed(applyPartialConversion(getModule(), target, patterns, &lowering.typeConverter))) {
      emitError(UnknownLoc::get(&getContext()), "Error lowering tile -> stripe\n");
      signalPassFailure();
    }

    getModule().walk([](FuncOp funcOp) {
      // Wraps function body with a ParallelForOp to represent Stripe's 'main' block.
      stripe::createMainParallelFor(funcOp);
    });
  }

  static std::unique_ptr<mlir::Pass> Create() { return std::make_unique<LoweringPass>(); }
};

OwningModuleRef LowerIntoStripe(ModuleOp workspace) {
  IVLOG(1, "LowerIntoStripe");
  OwningModuleRef module(llvm::cast<ModuleOp>(workspace.getOperation()->clone()));
  mlir::PassManager pm(workspace.getContext(), true);
  IVLOG(1, "before:\n" << mlir::debugString(*module->getOperation()));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(LoweringPass::Create());
  pm.addPass(mlir::createCanonicalizerPass());
  // pm.addPass(mlir::createCSEPass());
  auto result = pm.run(*module);
  if (failed(result)) {
    IVLOG(1, "LowerIntoStripe failed: " << mlir::debugString(*module->getOperation()));
    throw std::runtime_error("Lowering to stripe dialect failure");
  }
  IVLOG(1, "after:\n" << mlir::debugString(*module->getOperation()));
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

}  // namespace tile
}  // namespace dialect
}  // namespace pmlc
