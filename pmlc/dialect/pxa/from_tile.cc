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

using tile::Contraction;
using tile::ContractionOp;
using vertexai::tile::DataType;

using mlir::AffineLoadOp;
using mlir::AllocOp;
using mlir::ArrayRef;
using mlir::ConversionPattern;
using mlir::ConversionPatternRewriter;
using mlir::FloatType;
using mlir::FuncOp;
using mlir::FunctionType;
using mlir::IntegerType;
using mlir::MLIRContext;

using mlir::OpConversionPattern;
using mlir::Operation;
using mlir::PatternMatchResult;
using mlir::RankedTensorType;
using mlir::ReturnOp;
using mlir::Type;
using mlir::Value;

static stripe::TensorType convertIntoTensorType(Type type) {
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
    if (auto stype = type.dyn_cast<eltwise::ScalarType>()) {
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

struct ReturnOpConversion : public LoweringBase<ReturnOp> {
  explicit ReturnOpConversion(MLIRContext* ctx) : LoweringBase(ctx) {}

  void rewrite(ReturnOp op, ArrayRef<Value*> operands, ConversionPatternRewriter& rewriter) const override {
    std::cerr << "WHY!\n";
    rewriter.create<TerminateOp>(op.getLoc());
    rewriter.eraseOp(op);
  }
};

struct EltwiseOpConversion : public ConversionPattern {
  explicit EltwiseOpConversion(MLIRContext* ctx, StringRef opName) : ConversionPattern(opName, 1, ctx) {}
  PatternMatchResult match(Operation* op) const override { return this->matchSuccess(); }
  void rewrite(Operation* op, ArrayRef<Value*> operands, ConversionPatternRewriter& rewriter) const override {
    auto eop = mlir::cast<eltwise::EltwiseOp>(op);
    auto val = eop.buildStandard(operands, rewriter);
    rewriter.replaceOp(op, val);
  }
};

/*
static Value* MakeCombination(            //
    ConversionPatternRewriter* rewriter,  //
    Location loc,                         //
    CombinationKind combo,                //
    ScalarType scalarType,                //
    ArrayRef<Value*> operands) {
  switch (combo) {
    case CombinationKind::none:
      return operands[0];
    case CombinationKind::add:
      return rewriter->create<eltwise::AddOp>(loc, scalarType, operands).result();
    case CombinationKind::cond: {
      auto constOp = createInit(rewriter, loc, scalarType);
      auto cmpEqOp = rewriter->create<eltwise::CmpEqOp>(loc, scalarType, operands.drop_back());
      llvm::SmallVector<Value*, 3> args{cmpEqOp.result(), operands.back(), constOp.result()};
      return rewriter->create<eltwise::SelectOp>(loc, scalarType, args);
    }
    case CombinationKind::eq:
      return rewriter->create<eltwise::CmpEqOp>(loc, scalarType, operands).result();
    case CombinationKind::mul:
      return rewriter->create<eltwise::MulOp>(loc, scalarType, operands).result();
  }
  throw std::runtime_error("Invalid combination op");
}
*/

struct ContractionOpConversion : public LoweringBase<ContractionOp> {
  explicit ContractionOpConversion(MLIRContext* ctx) : LoweringBase(ctx) {}

  void rewrite(ContractionOp op, ArrayRef<Value*> operands, ConversionPatternRewriter& rewriter) const override {
    // Gather some basic info
    auto loc = op.getLoc();
    TypeConverter typeConverter(ctx);
    auto outType = typeConverter.convertType(op.result()->getType()).cast<MemRefType>();
    // Make an allocation for the output
    auto outRef = rewriter.create<AllocOp>(loc, outType).getResult();
    // Get the shape (TODO: fix this to not use stripe)
    auto resultTensorType = convertIntoTensorType(op.result()->getType());
    llvm::SmallVector<stripe::TensorType, 4> shapes{resultTensorType};
    for (auto src : op.operands()) {
      shapes.emplace_back(convertIntoTensorType(src->getType()));
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
    // Create the loads
    llvm::SmallVector<Value*, 4> vals;
    for (size_t i = 1; i < operands.size(); i++) {
      auto map = op.srcs().getValue()[i - 1].cast<AffineMapAttr>().getValue();
      vals.push_back(rewriter.create<AffineLoadOp>(loc, operands[i], map, idxs));
    }
    // TODO: Do the combination op
    Value* outVal = operands[0];
    // Create the store
    auto outMap = op.sink();
    rewriter.create<ReduceOp>(loc, op.agg(), outRef, outVal, outMap, idxs);

    rewriter.create<AffineTerminatorOp>(loc);
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
  patterns.insert<ReturnOpConversion>(&getContext());
  patterns.insert<ContractionOpConversion>(&getContext());
  auto eltwiseOps = util::getAllOpsWithInterface<eltwise::EltwiseOp>(&getContext());
  for (auto op : eltwiseOps) {
    patterns.insert<EltwiseOpConversion>(&getContext(), op->name);
  }

  // Run the conversion
  // if (failed(applyFullConversion(getModule(), target, patterns, nullptr))) {
  if (failed(applyPartialConversion(getModule(), target, patterns, nullptr))) {
    getModule().dump();
    emitError(mlir::UnknownLoc::get(&getContext()), "Error lowering tile -> pxa\n");
    signalPassFailure();
  }

  getModule().dump();
}

static mlir::PassRegistration<LoweringPass> legalize_pass(  //
    "tile-legalize-to-pxa",                                 //
    "Legalize from Tile dialect to PXA dialect");

}  // namespace pmlc::dialect::pxa
