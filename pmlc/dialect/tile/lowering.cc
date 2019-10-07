// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/lowering.h"

#include "llvm/Support/FormatVariadic.h"

#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "base/util/logging.h"
#include "pmlc/dialect/eltwise/dialect.h"
#include "pmlc/dialect/eltwise/ops.h"
#include "pmlc/dialect/eltwise/util.h"
#include "pmlc/dialect/stripe/dialect.h"
#include "pmlc/dialect/stripe/ops.h"
#include "pmlc/dialect/stripe/transcode.h"
#include "pmlc/dialect/tile/contraction.h"
#include "pmlc/dialect/tile/ops.h"
#include "pmlc/dialect/tile/program.h"

using mlir::AbstractOperation;
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

struct TypeConverter : public mlir::TypeConverter {
  using mlir::TypeConverter::convertType;

  Type convertType(Type type) override {
    IVLOG(2, "TypeConverter::convertType> " << type);
    if (auto funcType = type.dyn_cast<FunctionType>()) {
      IVLOG(4, "  FunctionType");
      return funcType;
    }
    if (auto rankedType = type.dyn_cast<RankedTensorType>()) {
      IVLOG(4, "  RankedTensorType");
      auto shape = rankedType.getShape();
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
      return stripe::TensorType::get(rankedType.getElementType(), newShape, stripe::OffsetsMap{}, false);
    }
    if (auto tensorType = type.dyn_cast<stripe::TensorType>()) {
      IVLOG(4, "  TensorType");
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

static Value* ConvertOutputTensor(FuncOp funcOp, Value* tensor) {
  for (auto user : tensor->getUsers()) {
    if (auto retOp = llvm::dyn_cast<ReturnOp>(user)) {
      for (unsigned i = 0; i < retOp.getNumOperands(); i++) {
        if (retOp.getOperand(i) == tensor) {
          auto inputs = funcOp.getAttrOfType<IntegerAttr>("inputs").getInt();
          return funcOp.getArgument(inputs + i);
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
    auto terminator = domainOp.body().front().getTerminator();
    auto contractionOp = llvm::dyn_cast<ContractionOp>(terminator);
    if (!contractionOp) {
      return matchFailure();
    }

    auto resultType = domainOp.result()->getType();
    auto resultTensorType = lowering->typeConverter.convertType(resultType).cast<stripe::TensorType>();
    IVLOG(3, "resultTensorType: " << resultTensorType);
    llvm::SmallVector<stripe::TensorType, 4> shapes{resultTensorType};
    for (auto src : contractionOp.getSourceIndexMaps()) {
      auto srcOp = llvm::cast<AffineSourceIndexMapOp>(src->getDefiningOp());
      auto srcType = srcOp.tensor()->getType();
      auto srcTensorType = lowering->typeConverter.convertType(srcType).cast<stripe::TensorType>();
      shapes.push_back(srcTensorType);
    }
    Contraction contraction{contractionOp};
    IndexBounds bounds;
    SimpleConstraints constraints;
    std::tie(bounds, constraints) = contraction.ComputeBounds(shapes);

    // add induction vars
    llvm::SmallVector<int64_t, 8> ranges;
    for (const auto& kvp : bounds) {
      uint64_t range = kvp.second.max - kvp.second.min + 1;
      ranges.emplace_back(range);
    }

    auto outputTensor = ConvertOutputTensor(op->getParentOfType<FuncOp>(), domainOp.result());
    if (!outputTensor) {
      auto allocOp = rewriter.create<stripe::AllocateOp>(op->getLoc(), resultTensorType);
      outputTensor = allocOp.result();
      Type tensorRefType = stripe::TensorRefType::get(resultTensorType);
      auto refOp = rewriter.create<stripe::TensorRefOp>(op->getLoc(), tensorRefType, outputTensor);
      outputTensor = refOp.result();
    }

    llvm::SmallVector<Value*, 4> tensors{outputTensor};
    for (auto src : contractionOp.getSourceIndexMaps()) {
      auto srcOp = llvm::cast<AffineSourceIndexMapOp>(src->getDefiningOp());
      tensors.emplace_back(srcOp.tensor());
    }

    auto aggKind = contractionOp.getAggregationKind();
    auto aggOpTag = llvm::formatv("agg_op_{0}", eltwise::stringifyAggregationKind(aggKind));
    auto comboKind = contractionOp.getCombinationKind();
    auto comboOpTag = llvm::formatv("combo_op_{0}", eltwise::stringifyCombinationKind(comboKind));
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

    unsigned argcnt = 0;
    stripe::SymbolValueMap idxs;
    for (const auto& kvp : bounds) {
      auto arg = body->addArgument(stripe::AffineType::get(rewriter.getContext()));
      auto argAttrs = llvm::SmallVector<NamedAttribute, 1>{
          {rewriter.getIdentifier("__name"), rewriter.getStringAttr(kvp.first)},
      };
      auto name = llvm::formatv("arg{0}", argcnt++);
      forOp.setAttr(name.str(), rewriter.getDictionaryAttr(argAttrs));
      idxs.emplace(kvp.first, arg);
    }

    // add constraints
    // TODO
    // for (const auto& constraint : constraints) {
    //   auto lhs = Integerize(constraint.poly, bounds);  // lhs <= rhs;
    //   lhs -= constraint.rhs;                           // lhs <= 0;
    //   lhs = -lhs;                                      // lhs >= 0
    //   kernel->constraints.emplace_back(lhs);
    // }

    // add refinements
    Value* output = nullptr;
    llvm::SmallVector<Value*, 4> inputs;
    for (unsigned i = 0; i < contraction.accesses.size(); i++) {
      const auto& access = contraction.accesses[i];
      llvm::SmallVector<Value*, 4> offsets;
      for (const auto& poly : access) {
        auto affine = Integerize(poly, bounds);
        offsets.emplace_back(stripe::AffineIntoMLIR(&rewriter, forOp.getOperation(), idxs, affine));
      }
      Type outType = tensors[i]->getType();
      if (!outType.isa<stripe::TensorRefType>()) {
        auto outTensorType = lowering->typeConverter.convertType(outType).cast<stripe::TensorType>();
        outType = stripe::TensorRefType::get(outTensorType);
      }
      auto tensorRefType = outType.cast<stripe::TensorRefType>();
      auto refineOp = rewriter.create<stripe::RefineOp>(op->getLoc(), outType, tensors[i], offsets);
      if (i) {
        auto refAttrs = llvm::SmallVector<NamedAttribute, 1>{
            {rewriter.getIdentifier("contraction"), rewriter.getUnitAttr()},
        };
        refineOp.setAttr(stripe::Dialect::getStripeAttrsName(), rewriter.getDictionaryAttr(refAttrs));
        // LOAD
        auto elementType = tensorRefType.getElementType();
        auto intoType = eltwise::GetTensorType(elementType);
        auto loadOp = rewriter.create<stripe::LoadOp>(op->getLoc(), intoType, refineOp.result());
        inputs.emplace_back(loadOp.into());
      } else {
        output = refineOp.result();
      }
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
    IVLOG(2, "EltwiseOpConversion::matchAndRewrite> " << *op);

    auto resultType = op->getResult(0)->getType();
    auto resultTensorType = lowering->typeConverter.convertType(resultType).cast<stripe::TensorType>();
    IVLOG(3, "resultTensorType: " << resultTensorType);

    auto outputTensor = ConvertOutputTensor(op->getParentOfType<FuncOp>(), op->getResult(0));
    if (!outputTensor) {
      auto allocOp = rewriter.create<stripe::AllocateOp>(op->getLoc(), resultTensorType);
      outputTensor = allocOp.result();
      Type tensorRefType = stripe::TensorRefType::get(resultTensorType);
      auto refOp = rewriter.create<stripe::TensorRefOp>(op->getLoc(), tensorRefType, outputTensor);
      outputTensor = refOp.result();
    }

    llvm::SmallVector<int64_t, 8> ranges;
    for (const auto& dim : resultTensorType.getShape()) {
      ranges.emplace_back(dim.size);
    }

    auto forOp = rewriter.create<stripe::ParallelForOp>(  //
        op->getLoc(),                                     //
        rewriter.getI64ArrayAttr(ranges));
    auto eltwiseOpTag = llvm::formatv("eltwise_{0}", eltwise::getOpName(op->getName()));
    auto attrs = llvm::SmallVector<NamedAttribute, 3>{
        {rewriter.getIdentifier("eltwise"), rewriter.getUnitAttr()},
        {rewriter.getIdentifier("kernel"), rewriter.getUnitAttr()},
        {rewriter.getIdentifier(eltwiseOpTag.str()), rewriter.getUnitAttr()},
        // {rewriter.getIdentifier(op->getName().getStringRef()), rewriter.getUnitAttr()},
    };
    forOp.setAttr(stripe::Dialect::getStripeAttrsName(), rewriter.getDictionaryAttr(attrs));
    auto body = rewriter.createBlock(&forOp.inner());

    stripe::SymbolValueMap idxs;
    const auto& dims = resultTensorType.getShape();
    for (unsigned i = 0; i < dims.size(); i++) {
      auto arg = body->addArgument(stripe::AffineType::get(rewriter.getContext()));
      auto idxName = llvm::formatv("i{0}", i);
      auto argAttrs = llvm::SmallVector<NamedAttribute, 1>{
          {rewriter.getIdentifier("__name"), rewriter.getStringAttr(idxName.str())},
      };
      auto argName = llvm::formatv("arg{0}", i);
      forOp.setAttr(argName.str(), rewriter.getDictionaryAttr(argAttrs));
      idxs.emplace(idxName, arg);
    }

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
      if (!operandType.isa<stripe::TensorRefType>()) {
        auto operandTensorType = lowering->typeConverter.convertType(operandType).cast<stripe::TensorType>();
        operandType = stripe::TensorRefType::get(operandTensorType);
      }

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
      auto intoType = eltwise::GetTensorType(elementType);
      // LOAD
      auto loadOp = rewriter.create<stripe::LoadOp>(op->getLoc(), intoType, refineOp.result());
      inputs.emplace_back(loadOp.into());
    }

    // INTRINSIC
    auto scalarType = resultTensorType.getElementType().cast<eltwise::ScalarType>();
    auto abstractOp = op->getAbstractOperation();
    auto builder = abstractOp->getInterface<eltwise::EltwiseBuilder>();
    if (!builder) {
      op->emitError("EltwiseBuilder expected for intrisnic");
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
    IVLOG(2, "FuncOpConversion::matchAndRewrite> " << type);

    // Convert the original function arguments.
    mlir::TypeConverter::SignatureConversion result(type.getNumInputs() + type.getNumResults());
    for (unsigned i = 0; i < type.getNumInputs(); ++i) {
      IVLOG(3, "  input(" << i << ")");
      if (failed(lowering->typeConverter.convertSignatureArg(i, type.getInput(i), result))) {
        return matchFailure();
      }
    }

    // append old results onto the end of new inputs
    for (unsigned i = 0; i < type.getNumResults(); i++) {
      auto argIndex = type.getNumInputs() + i;
      IVLOG(3, "  result(" << i << ")");
      if (failed(lowering->typeConverter.convertSignatureArg(argIndex, type.getResult(i), result))) {
        return matchFailure();
      }
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

    for (unsigned i = 0; i < type.getNumInputs() + type.getNumResults(); i++) {
      auto name = llvm::formatv("_X{0}", i);
      auto attrName = stripe::Dialect::getDialectAttrName("name");
      newFuncOp.setArgAttr(i, attrName, rewriter.getStringAttr(name.str()));
    }

    // Tell the rewriter to convert the region signature.
    rewriter.applySignatureConversion(&newFuncOp.getBody(), result);

    // finally cause the old func op to be erased
    rewriter.replaceOp(op, llvm::None);

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
        AffineDomainOpConversion,    //
        FuncOpConversion,            //
        ReturnOpConversion>(&lowering);
    auto eltwiseOps = eltwise::getAllOpsWithInterface<eltwise::EltwiseOp>(&getContext());
    for (auto op : eltwiseOps) {
      patterns.insert<EltwiseOpConversion>(&lowering, op->name);
    }
    if (failed(applyPartialConversion(getModule(), target, patterns, &lowering.typeConverter))) {
      emitError(UnknownLoc::get(&getContext()), "Error lowering tile -> stripe\n");
      signalPassFailure();
    }

    getModule().walk<FuncOp>([](FuncOp funcOp) {
      // wrap with a parallel_for to represent the 'main' block
      auto& region = funcOp.getBody();
      OpBuilder builder(region);
      auto body = region.begin();
      auto it = body->begin();
      auto forOp = builder.create<stripe::ParallelForOp>(funcOp.getLoc(), builder.getI64ArrayAttr({}));
      auto attrs = llvm::SmallVector<NamedAttribute, 1>{
          {builder.getIdentifier("main"), builder.getUnitAttr()},
      };
      forOp.setAttr(stripe::Dialect::getStripeAttrsName(), builder.getDictionaryAttr(attrs));
      forOp.setAttr("name", builder.getStringAttr("main"));
      auto block = builder.createBlock(&forOp.inner());
      block->getOperations().splice(block->getOperations().end(), body->getOperations(), it, body->end());

      // Inject TensorRefOp between each block argument and first usage
      builder.setInsertionPointToStart(&region.front());
      for (auto arg : funcOp.getArguments()) {
        Type tensorRefType = stripe::TensorRefType::get(arg->getType().cast<stripe::TensorType>());
        auto refOp = builder.create<stripe::TensorRefOp>(funcOp.getLoc(), tensorRefType, arg);
        arg->replaceAllUsesWith(refOp);
        refOp.setOperand(arg);
      }
      builder.setInsertionPointToEnd(&region.front());
      builder.create<stripe::TerminateOp>(funcOp.getLoc());
    });
  }

  static std::unique_ptr<mlir::Pass> Create() { return std::make_unique<LoweringPass>(); }
};

OwningModuleRef LowerIntoStripe(MLIRContext* context, TileProgram* program) {
  IVLOG(1, "LowerIntoStripe");
  OwningModuleRef module(llvm::cast<ModuleOp>(program->module->getOperation()->clone()));
  mlir::PassManager pm;
  IVLOG(1, "before:" << *module);
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(LoweringPass::Create());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  auto result = pm.run(*module);
  if (failed(result)) {
    IVLOG(1, "LowerIntoStripe failed: " << *module);
    throw std::runtime_error("Lowering to stripe dialect failure");
  }
  IVLOG(1, "after:" << *module);
  return module;
}

}  // namespace tile
}  // namespace dialect
}  // namespace pmlc
