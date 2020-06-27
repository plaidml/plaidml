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

static constexpr int64_t kUnusedDimension = -1;

static SmallVector<int64_t, 8> getFlattenedTileDimMapping(AffineMap map) {
  SmallVector<int64_t, 8> ret;
  for (const auto &expr : map.getResults()) {
    if (auto dimExpr = expr.dyn_cast<AffineDimExpr>()) {
      ret.push_back(dimExpr.getPosition());
    } else {
      ret.push_back(kUnusedDimension);
    }
  }
  return ret;
}

class XSMMGemmDispatchLowering : public OpRewritePattern<xsmm::GemmDispatchOp> {
public:
  using OpRewritePattern<xsmm::GemmDispatchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xsmm::GemmDispatchOp op,
                                PatternRewriter &rewriter) const override {
    Impl impl(op, rewriter);
    auto symbol = impl.getOrInsertDispatchFunc();
    SmallVector<Value, 6> args;
    for (auto i : impl.tileld) {
      args.push_back(impl.createConstantIntOp(i));
    }
    for (auto i : impl.tile) {
      args.push_back(impl.createConstantIntOp(i));
    }
    auto dispatch = rewriter.create<CallOp>(op.getLoc(), symbol,
                                            rewriter.getIntegerType(64), args);
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
      const char *symbol = "plaidml_rt_xsmm_dispatch_gemm_f32";
      auto context = module.getContext();
      if (module.lookupSymbol(symbol)) {
        return SymbolRefAttr::get(symbol, context);
      }
      OpBuilder builder(module.getBodyRegion());
      SmallVector<Type, 6> inputs{i32Type, i32Type, i32Type,
                                  i32Type, i32Type, i32Type};
      auto funcType = builder.getFunctionType(inputs, i64Type);
      ArrayRef<NamedAttribute> attrs{};
      builder.create<FuncOp>(loc, symbol, funcType, attrs);
      return SymbolRefAttr::get(symbol, context);
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
    SmallVector<Value, 6> args{a, b, c};
    args.push_back(op.ptr());

    rewriter.create<CallOp>(op.getLoc(), symbol, ArrayRef<Type>{}, args);
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
      const char *symbol = "plaidml_rt_xsmm_exec_gemm_f32";
      auto context = module.getContext();
      if (module.lookupSymbol(symbol)) {
        return SymbolRefAttr::get(symbol, context);
      }
      OpBuilder builder(module.getBodyRegion());
      SmallVector<Type, 4> inputs{unrankedType, unrankedType, unrankedType,
                                  i64Type};
      ArrayRef<Type> results{};
      auto funcType = builder.getFunctionType(inputs, results);
      ArrayRef<NamedAttribute> attrs{};
      builder.create<FuncOp>(loc, symbol, funcType, attrs);
      return SymbolRefAttr::get(symbol, context);
    }

    Value prepareOperand(Value operand, AffineMap accessMap,
                         ValueRange mapOperands, AffineMap tileMap,
                         ArrayRef<int64_t> sizes) {
      SmallVector<int64_t, 8> shape;
      auto flat = getFlattenedTileDimMapping(tileMap);
      for (auto dim : flat) {
        if (dim == kUnusedDimension)
          shape.push_back(1);
        else
          shape.push_back(sizes[dim]);
      }

      int64_t outerOffset;
      SmallVector<int64_t, 4> outerStrides;
      auto memRefType = operand.getType().cast<MemRefType>();
      getStridesAndOffset(memRefType, outerStrides, outerOffset);

      ArrayRef<Value> emptyValues{};
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
