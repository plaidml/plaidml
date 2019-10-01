// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/lowering.h"

#include <vector>

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
#include "pmlc/dialect/tile/internal.h"
#include "pmlc/dialect/tile/ops.h"

using mlir::Block;
using mlir::ConversionPattern;
using mlir::ConversionPatternRewriter;
using mlir::ConversionTarget;
using mlir::FuncOp;
using mlir::FunctionType;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::Operation;
using mlir::OwningModuleRef;
using mlir::PatternMatchResult;
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
      llvm::SmallVector<stripe::TensorDim, 4> newShape(shape.size());
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
  MLIRContext* context;
  TypeConverter typeConverter;
};

struct LoweringBase : public ConversionPattern {
  LoweringBase(                   //
      StringRef rootOpName,       //
      LoweringContext* lowering,  //
      mlir::PatternBenefit benefit = 1)
      : ConversionPattern(rootOpName, benefit, lowering->context), lowering(lowering) {}

 protected:
  LoweringContext* lowering;
};

struct AffineConstantOpConversion : public LoweringBase {
  explicit AffineConstantOpConversion(LoweringContext* lowering)  //
      : LoweringBase(AffineConstantOp::getOperationName(), lowering) {}

  PatternMatchResult matchAndRewrite(   //
      Operation* op,                    //
      llvm::ArrayRef<Value*> operands,  //
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
      : LoweringBase(mlir::ReturnOp::getOperationName(), lowering) {}

  PatternMatchResult matchAndRewrite(   //
      Operation* op,                    //
      llvm::ArrayRef<Value*> operands,  //
      ConversionPatternRewriter& rewriter) const override {
    IVLOG(2, "ReturnOpConversion::matchAndRewrite>");
    rewriter.create<stripe::TerminateOp>(op->getLoc());
    rewriter.replaceOp(op, llvm::None);
    return matchSuccess();
  }
};

static mlir::Value* ConvertOutputTensor(FuncOp funcOp, mlir::Value* tensor) {
  for (auto user : tensor->getUsers()) {
    if (auto retOp = llvm::dyn_cast<mlir::ReturnOp>(user)) {
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

struct AffineDomainOpConversion : public LoweringBase {
  explicit AffineDomainOpConversion(LoweringContext* lowering)  //
      : LoweringBase(AffineDomainOp::getOperationName(), lowering) {}

  PatternMatchResult matchAndRewrite(   //
      Operation* op,                    //
      llvm::ArrayRef<Value*> operands,  //
      ConversionPatternRewriter& rewriter) const override {
    // TODO
    IVLOG(2, "AffineDomainOpConversion::matchAndRewrite>");
    auto domainOp = llvm::cast<AffineDomainOp>(op);
    auto terminator = domainOp.body().front().getTerminator();
    auto contractionOp = llvm::dyn_cast<ContractionOp>(terminator);
    if (!contractionOp) {
      return matchFailure();
    }

    auto resultType = domainOp.result()->getType();
    auto outputTensorType = lowering->typeConverter.convertType(resultType).cast<stripe::TensorType>();
    IVLOG(1, "outputTensorType: " << outputTensorType);
    std::vector<stripe::TensorType> shapes{outputTensorType};
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
      outputTensor = rewriter.create<stripe::AllocateOp>(op->getLoc(), outputTensorType).result();
      Type tensorRefType = stripe::TensorRefType::get(outputTensorType);
      auto refOp = rewriter.create<stripe::TensorRefOp>(op->getLoc(), tensorRefType, outputTensor);
      outputTensor = refOp.result();
    }

    std::vector<Value*> tensors{outputTensor};
    for (auto src : contractionOp.getSourceIndexMaps()) {
      auto srcOp = llvm::cast<AffineSourceIndexMapOp>(src->getDefiningOp());
      tensors.emplace_back(srcOp.tensor());
    }

    auto forOp = rewriter.create<stripe::ParallelForOp>(  //
        op->getLoc(),                                     //
        rewriter.getI64ArrayAttr(ranges));
    auto body = rewriter.createBlock(&forOp.inner());

    unsigned argcnt = 0;
    stripe::SymbolValueMap idxs;
    for (const auto& kvp : bounds) {
      auto arg = body->addArgument(stripe::AffineType::get(rewriter.getContext()));
      std::vector<NamedAttribute> argAttrs;
      argAttrs.emplace_back(rewriter.getIdentifier("__name"), rewriter.getStringAttr(kvp.first));
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
    std::vector<mlir::Value*> locals;
    for (unsigned i = 0; i < contraction.accesses.size(); i++) {
      const auto& access = contraction.accesses[i];
      std::vector<Value*> offsets;
      for (const auto& poly : access) {
        auto affine = Integerize(poly, bounds);
        offsets.emplace_back(stripe::AffineIntoMLIR(&rewriter, idxs, affine));
      }
      Type outType = tensors[i]->getType();
      if (!outType.isa<stripe::TensorRefType>()) {
        auto outTensorType = lowering->typeConverter.convertType(outType).cast<stripe::TensorType>();
        outType = stripe::TensorRefType::get(outTensorType);
      }
      auto tensorRefType = outType.cast<stripe::TensorRefType>();
      auto refineOp = rewriter.create<stripe::RefineOp>(op->getLoc(), outType, tensors[i], offsets);
      if (i) {
        // LOAD
        auto elementType = tensorRefType.getElementType();
        auto intoType = eltwise::GetTensorType(elementType);
        auto loadOp = rewriter.create<stripe::LoadOp>(op->getLoc(), intoType, refineOp.result());
        locals.emplace_back(loadOp.into());
      } else {
        locals.emplace_back(refineOp.result());
      }
    }

    // Combination Operation
    // TODO
    auto scalarType = outputTensorType.getElementType().cast<eltwise::ScalarType>();
    auto comboOp = rewriter.create<eltwise::AddOp>(op->getLoc(), scalarType, locals[1], locals[2]);

    // STORE/Aggregate
    // TODO
    // rewriter.create<stripe::StoreOp>(op->getLoc(), locals[0], comboOp.result());
    auto aggType = symbolizeAggTypeEnum("add");
    auto aggInt = static_cast<int>(aggType.getValue());
    auto aggAttr = rewriter.getI64IntegerAttr(aggInt);
    rewriter.create<stripe::AggregateOp>(op->getLoc(), locals[0], comboOp.result(), aggAttr);

    rewriter.create<stripe::TerminateOp>(op->getLoc());
    rewriter.replaceOp(op, {outputTensor});

    // TODO: use_default
    // TODO: no_defract
    // TODO: NeedsInitialize

    return matchSuccess();
  }
};

struct FuncOpConversion : public LoweringBase {
  explicit FuncOpConversion(LoweringContext* lowering)  //
      : LoweringBase(FuncOp::getOperationName(), lowering) {}

  PatternMatchResult matchAndRewrite(  //
      Operation* op,                   //
      ArrayRef<Value*> operands,       //
      ConversionPatternRewriter& rewriter) const override {
    auto funcOp = llvm::cast<mlir::FuncOp>(op);
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
    std::vector<NamedAttribute> attrs{
        {rewriter.getIdentifier("program"), rewriter.getUnitAttr()},
    };
    newFuncOp.setAttr(stripe::Dialect::getStripeAttrsName(), rewriter.getDictionaryAttr(attrs));
    newFuncOp.setAttr("inputs", rewriter.getI32IntegerAttr(type.getNumInputs()));
    newFuncOp.setAttr("outputs", rewriter.getI32IntegerAttr(type.getNumResults()));

    for (unsigned i = 0; i < type.getNumInputs() + type.getNumResults(); i++) {
      auto name = llvm::formatv("X{0}", i);
      auto attrName = stripe::Dialect::getDialectAttrName(rewriter.getContext(), "name");
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
    target.addLegalDialect<eltwise::Dialect, stripe::Dialect>();
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
    if (failed(applyPartialConversion(getModule(), target, patterns, &lowering.typeConverter))) {
      emitError(mlir::UnknownLoc::get(&getContext()), "Error lowering tile -> stripe\n");
      signalPassFailure();
    }

    getModule().walk<FuncOp>([](FuncOp funcOp) {
      // wrap with a parallel_for to represent the 'main' block
      mlir::OpBuilder builder(funcOp.getBody());
      auto body = funcOp.getBody().begin();
      auto it = body->begin();
      auto forOp = builder.create<stripe::ParallelForOp>(funcOp.getLoc(), builder.getI64ArrayAttr({}));
      std::vector<NamedAttribute> attrs{
          {builder.getIdentifier("main"), builder.getUnitAttr()},
      };
      forOp.setAttr(stripe::Dialect::getStripeAttrsName(), builder.getDictionaryAttr(attrs));
      auto block = builder.createBlock(&forOp.inner());
      block->getOperations().splice(block->getOperations().end(), body->getOperations(), it, body->end());

      // Inject TensorRefOp between each block argument and first usage
      builder.setInsertionPointToStart(&funcOp.getBody().front());
      for (auto arg : funcOp.getArguments()) {
        Type tensorRefType = stripe::TensorRefType::get(arg->getType().cast<stripe::TensorType>());
        auto refOp = builder.create<stripe::TensorRefOp>(funcOp.getLoc(), tensorRefType, arg);
        arg->replaceAllUsesWith(refOp);
        refOp.setOperand(arg);
      }
      builder.setInsertionPointToEnd(&funcOp.getBody().front());
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
