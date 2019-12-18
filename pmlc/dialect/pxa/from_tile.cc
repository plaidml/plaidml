// Copyright 2019, Intel Corporation

#include "pmlc/dialect/pxa/from_tile.h"

#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/DialectConversion.h"

#include "base/util/logging.h"
#include "pmlc/dialect/eltwise/dialect.h"
#include "pmlc/dialect/eltwise/ops.h"
#include "pmlc/dialect/pxa/dialect.h"
#include "pmlc/dialect/pxa/ops.h"
#include "pmlc/dialect/tile/contraction.h"
#include "pmlc/dialect/tile/ops.h"
#include "pmlc/util/util.h"

namespace pmlc::dialect::pxa {

using eltwise::ScalarConstantOp;
using eltwise::ScalarType;
using tile::AggregationKind;
using tile::CombinationKind;
using tile::Contraction;
using tile::ContractionOp;
using vertexai::tile::DataType;

using mlir::AffineLoadOp;
using mlir::AllocOp;
using mlir::ArrayRef;
using mlir::ConversionPattern;
using mlir::ConversionPatternRewriter;
using mlir::FloatAttr;
using mlir::FloatType;
using mlir::FuncOp;
using mlir::FunctionType;
using mlir::IntegerAttr;
using mlir::IntegerType;
using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::OpConversionPattern;
using mlir::Operation;
using mlir::PatternMatchResult;
using mlir::RankedTensorType;
using mlir::ReturnOp;
using mlir::Type;
using mlir::Value;

static stripe::TensorType ConvertIntoTensorType(Type type) {
  auto rankedTensorType = eltwise::getRankedTensorType(type);
  auto shape = rankedTensorType.getShape();
  auto cls = mlir::Identifier::get(stripe::kAddressClassIdentifier, type.getContext());
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

struct LoweringPass : public mlir::ModulePass<LoweringPass> {
  void runOnModule() override;
};

struct TypeConverter : public mlir::TypeConverter {
  using mlir::TypeConverter::convertType;
  MLIRContext* ctx;
  explicit TypeConverter(MLIRContext* ctx) : ctx(ctx) {}

  Type convertType(Type type) override {
    IVLOG(2, "TypeConverter::convertType> " << mlir::debugString(type));
    if (type.isa<FunctionType>()) {
      IVLOG(4, "  FunctionType");
      return type;
    }
    if (auto stype = type.dyn_cast<ScalarType>()) {
      return stype.toStandard();
    }
    if (auto rtype = type.dyn_cast<RankedTensorType>()) {
      IVLOG(4, "  RankedTensorType");
      return MemRefType::get(rtype.getShape(), convertType(rtype.getElementType()));
    }
    return {};
  }
};

template <typename OpType>
struct LoweringBase : public OpConversionPattern<OpType> {
  MLIRContext* ctx;

  explicit LoweringBase(MLIRContext* ctx) : OpConversionPattern<OpType>(ctx), ctx(ctx) {}
  PatternMatchResult match(Operation* op) const override { return this->matchSuccess(); }
};

struct FuncOpConversion : public LoweringBase<FuncOp> {
  explicit FuncOpConversion(MLIRContext* ctx) : LoweringBase(ctx) {}

  void rewrite(FuncOp op, ArrayRef<Value*> operands, ConversionPatternRewriter& rewriter) const override {
    FunctionType type = op.getType();
    IVLOG(2, "FuncOpConversion::rewrite> " << mlir::debugString(type));

    // Convert the function signature
    TypeConverter typeConverter(ctx);
    mlir::TypeConverter::SignatureConversion result(type.getNumInputs() + type.getNumResults());
    for (unsigned i = 0; i < type.getNumInputs(); ++i) {
      result.addInputs(i, {typeConverter.convertType(type.getInput(i))});
    }
    for (unsigned i = 0; i < type.getNumResults(); ++i) {
      result.addInputs({typeConverter.convertType(type.getResult(i))});
    }

    // Create a new function with an updated signature.
    auto newOp = rewriter.cloneWithoutRegions(op);
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(), newOp.end());
    newOp.setType(FunctionType::get(result.getConvertedTypes(), llvm::None, op.getContext()));

    // Tell the rewriter to convert the region signature.
    rewriter.applySignatureConversion(&newOp.getBody(), result);

    // Finally cause the old func op to be erased
    rewriter.eraseOp(op);
  }
};

struct ScalarConstantOpConversion : public LoweringBase<ScalarConstantOp> {
  explicit ScalarConstantOpConversion(MLIRContext* ctx) : LoweringBase(ctx) {}

  void rewrite(ScalarConstantOp op, ArrayRef<Value*> operands, ConversionPatternRewriter& rewriter) const override {
    Type type = op.getType().dyn_cast<ScalarType>().toStandard();
    Attribute val = op.getValue();
    if (auto ftype = type.dyn_cast<FloatType>()) {
      auto fattr = val.cast<FloatAttr>();
      val = FloatAttr::get(ftype, fattr.getValueAsDouble());
    } else if (auto itype = type.dyn_cast<IntegerType>()) {
      auto iattr = val.cast<IntegerAttr>();
      val = IntegerAttr::get(itype, iattr.getInt());
    } else {
      llvm_unreachable("Invalid scalar constant op");
    }
    auto newOp = rewriter.create<mlir::ConstantOp>(op.getLoc(), type, val);
    rewriter.replaceOp(op, {newOp});
  }
};

struct EltwiseOpConversion : public ConversionPattern {
  explicit EltwiseOpConversion(MLIRContext* ctx, StringRef opName) : ConversionPattern(opName, 1, ctx) {}
  PatternMatchResult match(Operation* op) const override { return this->matchSuccess(); }
  void rewrite(Operation* op, ArrayRef<Value*> operands, ConversionPatternRewriter& rewriter) const override {
    /*
    auto eop = mlir::cast<eltwise::EltwiseOp>(op);
    auto val = eop.buildStandard(rewriter, op-> operands, rewriter);
    rewriter.replaceOp(op, val);
    */
  }
};

static ScalarType GetScalarType(Value* val) {
  auto t = val->getType();
  if (auto ttype = t.dyn_cast<mlir::TensorType>()) {
    t = ttype.getElementType();
  }
  return t.cast<ScalarType>();
}

static Value* Cast(OpBuilder& builder, ScalarType in, ScalarType out, Value* stdVal) {
  TypeConverter typeConverter(stdVal->getContext());
  auto stypeOut = typeConverter.convertType(out);
  if (stypeOut == stdVal->getType()) {
    return stdVal;
  }
  bool in_signed = !is_uint(in.type());
  bool out_signed = !is_uint(out.type());
  return builder.create<CastOp>(stdVal->getLoc(), stypeOut, stdVal, builder.getBoolAttr(in_signed),
                                builder.getBoolAttr(out_signed));
}

static ScalarType CommonSupertype(ScalarType a, ScalarType b) {
  return ScalarType::get(a.getContext(), CommonSupertype(a.type(), b.type()));
}

static Value* CreateInit(OpBuilder& builder, Location loc, ScalarType stype, AggregationKind agg) {
  Type type = stype.toStandard();
  if (auto ftype = type.dyn_cast<FloatType>()) {
    switch (agg) {
      case AggregationKind::add:
        return builder.create<mlir::ConstantOp>(loc, type, FloatAttr::get(ftype, 0.0));
      case AggregationKind::mul:
        return builder.create<mlir::ConstantOp>(loc, type, FloatAttr::get(ftype, 1.0));
      default:
        llvm_unreachable("Unsupported aggregation for CreateInit");
    }
  } else if (auto itype = type.dyn_cast<IntegerType>()) {
    switch (agg) {
      case AggregationKind::add:
        return builder.create<mlir::ConstantOp>(loc, type, IntegerAttr::get(itype, 0));
      case AggregationKind::mul:
        return builder.create<mlir::ConstantOp>(loc, type, IntegerAttr::get(itype, 1));
      default:
        llvm_unreachable("Unsupported aggregation for CreateInit");
    }
  }
  llvm_unreachable("Unknown type for CreateInit");
}

static Value* MakeCombination(            //
    ConversionPatternRewriter& rewriter,  //
    Location loc,                         //
    CombinationKind combo,                //
    ArrayRef<ScalarType> types,           //
    ArrayRef<Value*> vals)                //
{
  if (combo == CombinationKind::none) {
    return vals[0];
  } else if (combo == CombinationKind::cond) {
    auto ctype = CommonSupertype(types[0], types[1]);
    auto v0 = Cast(rewriter, types[0], ctype, vals[0]);
    auto v1 = Cast(rewriter, types[1], ctype, vals[1]);
    auto cmp = eltwise::CmpEqOp::buildStandard(rewriter, loc, ctype, {v0, v1});
    auto zero = CreateInit(rewriter, loc, types[2], AggregationKind::add);
    return eltwise::SelectOp::buildStandard(rewriter, loc, types[2], {cmp, vals[2], zero});
  } else {
    auto ctype = CommonSupertype(types[0], types[1]);
    auto v0 = Cast(rewriter, types[0], ctype, vals[0]);
    auto v1 = Cast(rewriter, types[1], ctype, vals[1]);
    switch (combo) {
      case CombinationKind::add:
        return eltwise::AddOp::buildStandard(rewriter, loc, ctype, {v0, v1});
      case CombinationKind::mul:
        return eltwise::MulOp::buildStandard(rewriter, loc, ctype, {v0, v1});
      case CombinationKind::eq:
        return eltwise::CmpEqOp::buildStandard(rewriter, loc, ctype, {v0, v1});
      default:
        llvm_unreachable("Invalid combination");
    }
  }
}

struct ContractionOpConversion : public LoweringBase<ContractionOp> {
  explicit ContractionOpConversion(MLIRContext* ctx) : LoweringBase(ctx) {}

  void rewrite(ContractionOp op, ArrayRef<Value*> operands, ConversionPatternRewriter& rewriter) const override {
    tile::ContractionOpOperandAdaptor new_op(operands);
    // Gather some basic info
    auto loc = op.getLoc();
    TypeConverter typeConverter(ctx);
    auto outType = typeConverter.convertType(op.result()->getType()).cast<MemRefType>();
    // Make an allocation for the output
    auto outRef = rewriter.create<AllocOp>(loc, outType).getResult();
    // Get the shape (TODO: fix this to not use stripe)
    auto resultTensorType = ConvertIntoTensorType(op.result()->getType());
    llvm::SmallVector<stripe::TensorType, 4> shapes{resultTensorType};
    for (auto src : op.operands()) {
      shapes.emplace_back(ConvertIntoTensorType(src->getType()));
    }
    // Do the actual maths
    Contraction contraction{op};
    bool no_reduce = op.no_reduce().hasValue();
    const auto& [bounds, constraints] = contraction.ComputeBounds(shapes, no_reduce);
    // Extract ranges
    llvm::SmallVector<int64_t, 8> ranges;
    for (const auto& [key, value] : bounds) {
      uint64_t range = value.max - value.min + 1;
      ranges.emplace_back(range);
    }
    // Make the outer loops
    auto runtime_ranges = ArrayRef<Value*>();
    auto forOp = rewriter.create<AffineParallelForOp>(loc, rewriter.getI64ArrayAttr(ranges), runtime_ranges);
    auto body = rewriter.createBlock(&forOp.inner());
    llvm::SmallVector<Value*, 8> idxs;
    for (size_t i = 0; i < ranges.size(); i++) {
      idxs.push_back(body->addArgument(rewriter.getIndexType()));
    }
    // Create the loads + casts
    llvm::SmallVector<Value*, 4> vals;
    llvm::SmallVector<ScalarType, 4> types;
    for (size_t i = 0; i < op.srcs().getValue().size(); i++) {
      auto map = op.srcs().getValue()[i].cast<AffineMapAttr>().getValue();
      auto new_in = *std::next(new_op.operands().begin(), i);
      vals.push_back(rewriter.create<AffineLoadOp>(loc, new_in, map, idxs));
      auto old_in = *std::next(op.operands().begin(), i);
      types.push_back(GetScalarType(old_in));
    }
    // TODO: Do the combination op
    Value* outVal = MakeCombination(rewriter, op.getLoc(), op.combo(), types, vals);
    // Create the store
    auto outMap = op.sink();
    rewriter.create<ReduceOp>(loc, op.agg(), outRef, outVal, outMap, idxs);
    rewriter.replaceOp(op, outRef);
  }
};

void LoweringPass::runOnModule() {
  // Set up target (i.e. what is legal)
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::AffineOpsDialect>();
  target.addLegalDialect<mlir::StandardOpsDialect>();
  target.addLegalDialect<Dialect>();
  target.addLegalOp<mlir::ModuleOp>();
  target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
    auto funcType = op.getType();
    if (funcType.getNumResults()) {
      return false;
    }
    // return typeConverter.isSignatureLegal(funcType);
    return true;
  });

  // Setup rewrite patterns
  mlir::OwningRewritePatternList patterns;
  patterns.insert<FuncOpConversion>(&getContext());
  patterns.insert<ContractionOpConversion>(&getContext());
  patterns.insert<ContractionOpConversion>(&getContext());
  patterns.insert<ScalarConstantOpConversion>(&getContext());
  auto eltwiseOps = util::getAllOpsWithInterface<eltwise::EltwiseOp>(&getContext());
  for (auto op : eltwiseOps) {
    patterns.insert<EltwiseOpConversion>(&getContext(), op->name);
  }

  // Run the conversion
  if (failed(applyPartialConversion(getModule(), target, patterns, nullptr))) {
    getModule().dump();
    emitError(mlir::UnknownLoc::get(&getContext()), "Error lowering tile -> pxa\n");
    signalPassFailure();
  }

  // Do a final fixup to remove returns
  getModule().walk([](FuncOp op) {
    mlir::Block& block = op.getBody().front();
    auto ret = mlir::cast<ReturnOp>(block.back());
    size_t cur_arg = op.getType().getNumInputs() - ret.getNumOperands();
    for (Value* v : ret.getOperands()) {
      auto defOp = v->getDefiningOp();
      v->replaceAllUsesWith(block.getArgument(cur_arg++));
      defOp->erase();
    }
    ret.erase();
  });

  // Do the final
}

static mlir::PassRegistration<LoweringPass> legalize_pass(  //
    "tile-legalize-to-pxa",                                 //
    "Legalize from Tile dialect to PXA dialect");

}  // namespace pmlc::dialect::pxa
