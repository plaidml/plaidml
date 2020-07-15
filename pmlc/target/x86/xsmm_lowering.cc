// Copyright 2020 Intel Corporation

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/xsmm/ir/ops.h"
#include "pmlc/target/x86/pass_detail.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::target::x86 {

namespace xsmm = dialect::xsmm;

namespace {

const char *kGemmInvokeF32 = "plaidml_rt_xsmm_gemm_invoke_f32";
const char *kGemmDispatchF32 = "plaidml_rt_xsmm_gemm_dispatch_f32";

class XSMMGemmDispatchLowering : public OpRewritePattern<xsmm::GemmDispatchOp> {
public:
  using OpRewritePattern<xsmm::GemmDispatchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xsmm::GemmDispatchOp op,
                                PatternRewriter &rewriter) const override {
    Impl impl(op, rewriter);
    auto symbol = impl.getOrInsertDispatchFunc();
    SmallVector<Value, 6> args;
    for (auto value : impl.tileld) {
      args.push_back(impl.createConstantIntOp(value));
    }
    for (auto value : impl.tile) {
      args.push_back(impl.createConstantIntOp(value));
    }
    auto dispatch =
        rewriter.create<CallOp>(op.getLoc(), symbol, impl.i64Type, args);
    rewriter.replaceOp(op, dispatch.getResult(0));
    return success();
  }

  struct Impl {
    xsmm::GemmDispatchOp op;
    PatternRewriter &rewriter;
    Location loc;
    ModuleOp module;
    Type i32Type;
    Type i64Type;
    SmallVector<unsigned, 3> tile;
    SmallVector<unsigned, 3> tileld;

    Impl(xsmm::GemmDispatchOp op, PatternRewriter &rewriter)
        : op(op), rewriter(rewriter), loc(op.getLoc()),
          module(op.getParentOfType<ModuleOp>()),
          i32Type(rewriter.getIntegerType(32)),
          i64Type(rewriter.getIntegerType(64)) {
      for (auto attr : op.tile().getValue()) {
        tile.push_back(attr.cast<IntegerAttr>().getInt());
      }
      for (auto attr : op.tileld().getValue()) {
        tileld.push_back(attr.cast<IntegerAttr>().getInt());
      }
    }

    FlatSymbolRefAttr getOrInsertDispatchFunc() {
      auto context = module.getContext();
      if (module.lookupSymbol(kGemmDispatchF32)) {
        return SymbolRefAttr::get(kGemmDispatchF32, context);
      }
      OpBuilder builder(module.getBodyRegion());
      SmallVector<Type, 6> inputs{i32Type, i32Type, i32Type,
                                  i32Type, i32Type, i32Type};
      auto funcType = builder.getFunctionType(inputs, i64Type);
      ArrayRef<NamedAttribute> attrs{};
      builder.create<FuncOp>(loc, kGemmDispatchF32, funcType, attrs);
      return SymbolRefAttr::get(kGemmDispatchF32, context);
    }

    Value createConstantIntOp(int64_t value) {
      return rewriter.create<ConstantIntOp>(loc, value, i32Type);
    }
  };
};

class XSMMGemmInvokeLowering : public OpRewritePattern<xsmm::GemmInvokeOp> {
public:
  using OpRewritePattern<xsmm::GemmInvokeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xsmm::GemmInvokeOp op,
                                PatternRewriter &rewriter) const override {
    Impl impl(op, rewriter);
    auto &tile = impl.tile;

    auto symbol = impl.getOrInsertInvokeFunc();
    auto a = impl.prepareOperand(op.a(), op.aAccessMap(), op.getOperandsForA(),
                                 op.aTileMap(), {tile[0], tile[2]});
    auto b = impl.prepareOperand(op.b(), op.bAccessMap(), op.getOperandsForB(),
                                 op.bTileMap(), {tile[2], tile[1]});
    auto c = impl.prepareOperand(op.c(), op.cAccessMap(), op.getOperandsForC(),
                                 op.cTileMap(), {tile[0], tile[1]});

    rewriter.create<CallOp>(op.getLoc(), symbol, ArrayRef<Type>{},
                            ArrayRef<Value>{a, b, c, op.ptr()});
    rewriter.replaceOp(op, op.c());
    return success();
  }

  struct Impl {
    xsmm::GemmInvokeOp op;
    PatternRewriter &rewriter;
    Location loc;
    ModuleOp module;
    Type i64Type;
    Type elementType;
    UnrankedMemRefType unrankedType;
    SmallVector<unsigned, 3> tile;

    Impl(xsmm::GemmInvokeOp op, PatternRewriter &rewriter)
        : op(op), rewriter(rewriter), loc(op.getLoc()),
          module(op.getParentOfType<ModuleOp>()),
          i64Type(rewriter.getIntegerType(64)),
          elementType(rewriter.getF32Type()),
          unrankedType(
              UnrankedMemRefType::get(elementType, /*memorySpace=*/0)) {
      for (auto attr : op.tile().getValue()) {
        tile.push_back(attr.cast<IntegerAttr>().getInt());
      }
    }

    FlatSymbolRefAttr getOrInsertInvokeFunc() {
      auto context = module.getContext();
      if (module.lookupSymbol(kGemmInvokeF32)) {
        return SymbolRefAttr::get(kGemmInvokeF32, context);
      }
      OpBuilder builder(module.getBodyRegion());
      SmallVector<Type, 4> inputs{unrankedType, unrankedType, unrankedType,
                                  i64Type};
      ArrayRef<Type> results{};
      // Insert a function attribute that will trigger the emission of the
      // corresponding `_mlir_ciface_xxx` interface so that external libraries
      // see a normalized ABI. This interface is added during std to llvm
      // conversion.
      ArrayRef<NamedAttribute> attrs{
          // builder.getNamedAttr("llvm.emit_c_interface",
          // builder.getUnitAttr()),
      };
      auto funcType = builder.getFunctionType(inputs, results);
      builder.create<FuncOp>(loc, kGemmInvokeF32, funcType, attrs);

      return SymbolRefAttr::get(kGemmInvokeF32, context);
    }

    Value prepareOperand(Value operand, AffineMap accessMap,
                         ValueRange mapOperands, AffineMap tileMap,
                         ArrayRef<int64_t> sizes) {
      SmallVector<int64_t, 8> shape;
      for (const auto &expr : tileMap.getResults()) {
        if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
          auto dim = dimExpr.getPosition();
          shape.push_back(sizes[dim]);
        } else {
          shape.push_back(1);
        }
      }

      ArrayRef<Value> emptyValues{};
      auto memRefType = operand.getType().cast<MemRefType>();
      SmallVector<int64_t, 4> staticOffsets(
          memRefType.getRank(), MemRefType::getDynamicStrideOrOffset());
      SmallVector<int64_t, 4> staticStrides(memRefType.getRank(), 1);
      auto offsets = expandAffineMap(rewriter, loc, accessMap, mapOperands);
      auto subview = rewriter.create<SubViewOp>(loc,
                                                /*source=*/operand,
                                                /*staticOffsets=*/staticOffsets,
                                                /*staticSizes=*/shape,
                                                /*staticStrides=*/staticStrides,
                                                /*offsets=*/*offsets,
                                                /*sizes=*/emptyValues,
                                                /*strides=*/emptyValues);
      return rewriter.create<MemRefCastOp>(loc, subview, unrankedType);
    }
  };
};

class LowerXSMMPass : public XSMMLoweringBase<LowerXSMMPass> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    patterns.insert<XSMMGemmInvokeLowering, XSMMGemmDispatchLowering>(
        &getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<AffineDialect, StandardOpsDialect>();
    target.addIllegalDialect<xsmm::XSMMDialect>();
    if (failed(applyPartialConversion(getFunction(), target, patterns)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createXSMMLoweringPass() {
  return std::make_unique<LowerXSMMPass>();
}

} // namespace pmlc::target::x86
