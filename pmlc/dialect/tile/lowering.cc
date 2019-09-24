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
#include "pmlc/dialect/eltwise/util.h"
#include "pmlc/dialect/stripe/dialect.h"
#include "pmlc/dialect/stripe/ops.h"
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
    IVLOG(1, "TypeConverter::convertType> " << type);
    if (auto funcType = type.dyn_cast<FunctionType>()) {
      IVLOG(1, "  FunctionType");
      return funcType;
    }
    if (auto rankedType = type.dyn_cast<RankedTensorType>()) {
      IVLOG(1, "  RankedTensorType");
      auto shape = rankedType.getShape();
      llvm::SmallVector<stripe::TensorDim, 4> newShape(shape.size());
      // TODO: instead of using natural strides, use the I/O map supplied by the user
      int64_t stride = 1;
      for (int i = shape.size() - 1; i >= 0; i--) {
        newShape[i].stride = stride;
        newShape[i].size = shape[i];
        stride *= shape[i];
      }
      return stripe::TensorType::get(rankedType.getElementType(), newShape);
    }
    if (auto tensorType = type.dyn_cast<stripe::TensorType>()) {
      IVLOG(1, "  TensorType");
      return type;
    }
    // if (auto indexType = type.dyn_cast<IndexType>()) {
    // }
    // return type;
    return {};
  }

  /// Materialize a conversion to allow for partial lowering of types.
  // Operation* materializeConversion(  //
  //     PatternRewriter& rewriter,     //
  //     Type resultType,               //
  //     ArrayRef<Value*> inputs,       //
  //     Location loc) override {
  //   assert(inputs.size() == 1 && "expected only one input value");
  //   return rewriter.create<toy::TypeCastOp>(loc, inputs[0], resultType);
  // }
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
    IVLOG(1, "AffineConstantOpConversion::matchAndRewrite>");
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
    IVLOG(1, "ReturnOpConversion::matchAndRewrite>");
    rewriter.create<stripe::TerminateOp>(op->getLoc());
    rewriter.replaceOp(op, llvm::None);
    return matchSuccess();
  }
};

struct AffineDomainOpConversion : public LoweringBase {
  explicit AffineDomainOpConversion(LoweringContext* lowering)  //
      : LoweringBase(AffineDomainOp::getOperationName(), lowering) {}

  PatternMatchResult matchAndRewrite(   //
      Operation* op,                    //
      llvm::ArrayRef<Value*> operands,  //
      ConversionPatternRewriter& rewriter) const override {
    // TODO
    IVLOG(1, "AffineDomainOpConversion::matchAndRewrite>");
    auto domainOp = llvm::cast<AffineDomainOp>(op);
    auto terminator = domainOp.body().front().getTerminator();
    auto contractionOp = llvm::dyn_cast<ContractionOp>(terminator);
    if (!contractionOp) {
      return matchFailure();
    }

    auto resultType = domainOp.result()->getType();
    auto outputTensorType = lowering->typeConverter.convertType(resultType).cast<stripe::TensorType>();
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

    Operation* retOp = nullptr;
    for (auto user : domainOp.result()->getUsers()) {
      if (llvm::isa<mlir::ReturnOp>(user)) {
        retOp = user;
      }
    }

    Value* newValue;
    if (retOp) {
      // Create a dummy op to satisfy verification; this will get killed as dead
      auto dummyOp = rewriter.create<stripe::AffineConstOp>(  //
          op->getLoc(),                                       //
          rewriter.getType<stripe::AffineType>(),             //
          rewriter.getI64IntegerAttr(0));
      newValue = dummyOp.result();
    } else {
      auto allocOp = rewriter.create<stripe::AllocateOp>(op->getLoc(), outputTensorType);
      newValue = allocOp.result();
    }

    std::vector<NamedAttribute> attrs;
    auto forOp = rewriter.create<stripe::ParallelForOp>(  //
        op->getLoc(),                                     //
        rewriter.getI64ArrayAttr(ranges),                 //
        rewriter.getDictionaryAttr(attrs));
    auto body = rewriter.createBlock(&forOp.inner());

    unsigned argcnt = 0;
    for (const auto& kvp : bounds) {
      body->addArgument(stripe::AffineType::get(rewriter.getContext()));
      std::vector<NamedAttribute> argAttrs;
      argAttrs.emplace_back(rewriter.getIdentifier("name"), rewriter.getStringAttr(kvp.first));
      // TODO: set index attrs
      auto name = llvm::formatv("arg{0}", argcnt++);
      forOp.setAttr(name.str(), rewriter.getDictionaryAttr(argAttrs));
    }

    // add refinements
    // for (auto src : contractionOp.getSourceIndexMaps()) {
    //   std::vector<Value*> offsets;
    //   auto dictAttrs = rewriter.getDictionaryAttr(attrs);
    //   rewriter.create<stripe::RefineOp>(op->getLoc(), src->getType(), src, offsets, dictAttrs);
    // }

    // add constraints
    // for (const auto& constraint : constraints) {
    //   auto lhs = Integerize(constraint.poly, bounds);  // lhs <= rhs;
    //   lhs -= constraint.rhs;                           // lhs <= 0;
    //   lhs = -lhs;                                      // lhs >= 0
    //   kernel->constraints.emplace_back(lhs);
    // }

    // LOAD
    // Combination Operation
    // STORE/Aggregate

    rewriter.create<stripe::TerminateOp>(op->getLoc());
    rewriter.replaceOp(op, {newValue});

    // if (NeedsInitialize(*kernel, out_ref_name, shapes[0])) {
    //   auto stmt = InitBuffer(main, op, shapes[0]);
    //   main->stmts.insert(std::prev(main->stmts.end()), stmt);
    //   auto ref_it = kernel->ref_by_into(out_ref_name);
    //   ref_it->mut().dir = RefDir::InOut;
    //   ref_it->mut().set_tag("initialized");
    // }

    // cion.agg_op = expr.agg_op;
    // cion.comb_op = expr.combo_op;
    // cion.no_defract = expr.no_defract;
    // if (expr.use_default) {
    //   cion.use_default = safe_at(&eval_.names_by_expr, expr.use_default.get());
    // }
    // This first spec represents the output
    // cion.specs.emplace_back(TensorSpec{});
    // std::vector<std::string> inputs;
    // for (const auto& src : expr.srcs) {
    //   TensorSpec spec;
    //   spec.id = safe_at(&eval_.names_by_expr, src->ref.get());
    //   inputs.push_back(spec.id);
    //   for (const auto& idx : src->idxs) {
    //     auto poly = idx->Accept(&poly_eval);
    //     spec.spec.push_back(poly);
    //   }
    //   cion.specs.emplace_back(spec);
    // }
    // for (const auto& idx : expr.sink_idxs->idxs) {
    //   auto poly = idx->Accept(&poly_eval);
    //   cion.specs[0].spec.push_back(poly);
    // }
    // for (const auto& size_expr : expr.sink_dims->dims) {
    //   auto size = size_expr->Accept(&dim_eval);
    //   cion.output_size.push_back(std::to_string(size));
    // }
    // for (const auto& constraint : expr.constraints) {
    //   auto poly = constraint->lhs->Accept(&poly_eval);
    //   auto range = constraint->rhs->Accept(&dim_eval);
    //   tile::math::RangeConstraint bound(poly, range);
    //   cion.constraints.emplace_back(bound);
    // }
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
    IVLOG(1, "FuncOpConversion::matchAndRewrite> " << type);

    // Convert the original function arguments.
    mlir::TypeConverter::SignatureConversion result(type.getNumInputs() + type.getNumResults());
    for (unsigned i = 0; i < type.getNumInputs(); ++i) {
      IVLOG(1, "  input(" << i << ")");
      if (failed(lowering->typeConverter.convertSignatureArg(i, type.getInput(i), result))) {
        return matchFailure();
      }
    }

    // append old results onto the end of new inputs
    for (unsigned i = 0; i < type.getNumResults(); i++) {
      auto argIndex = type.getNumInputs() + i;
      IVLOG(1, "  result(" << i << ")");
      if (failed(lowering->typeConverter.convertSignatureArg(argIndex, type.getResult(i), result))) {
        return matchFailure();
      }
    }

    // Create a new function with an updated signature.
    auto newFuncOp = rewriter.cloneWithoutRegions(funcOp);
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(), newFuncOp.end());
    newFuncOp.setType(FunctionType::get(result.getConvertedTypes(), llvm::None, funcOp.getContext()));

    auto prefix = llvm::formatv("{0}.", stripe::Dialect::getDialectNamespace());
    for (unsigned i = 0; i < type.getNumInputs() + type.getNumResults(); i++) {
      auto name = llvm::formatv("X{0}", i);
      newFuncOp.setArgAttr(i, prefix.str() + "name", rewriter.getStringAttr(name.str()));
    }

    // Tell the rewriter to convert the region signature.
    rewriter.applySignatureConversion(&newFuncOp.getBody(), result);
    rewriter.replaceOp(op, llvm::None);
    return matchSuccess();
  }
};

struct LoweringPass : public mlir::ModulePass<LoweringPass> {
  void runOnModule() override {
    LoweringContext lowering{&getContext()};
    ConversionTarget target(getContext());
    target.addLegalDialect<eltwise::Dialect, stripe::Dialect>();
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {  //
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
    mlir::populateFuncOpTypeConversionPattern(patterns, &getContext(), lowering.typeConverter);
    if (failed(applyPartialConversion(getModule(), target, patterns, &lowering.typeConverter))) {
      emitError(mlir::UnknownLoc::get(&getContext()), "Error lowering tile -> stripe\n");
      signalPassFailure();
    }
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
