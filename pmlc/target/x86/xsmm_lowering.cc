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

class XSMMGemmLowering : public OpRewritePattern<xsmm::GemmOp> {
public:
  using OpRewritePattern<xsmm::GemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xsmm::GemmOp op,
                                PatternRewriter &rewriter) const override {
    Impl impl(op, rewriter);
    auto &tile = impl.tile;

    auto symbola = impl.getOrInsertDispatchFunc();
    auto symbolb = impl.getOrInsertExecFunc();
    auto a = impl.prepareOperand(op.a(), op.aAccessMap(), op.getOperandsForA(),
                                 op.aTileMap(), {tile[0], tile[2]});
    auto b = impl.prepareOperand(op.b(), op.bAccessMap(), op.getOperandsForB(),
                                 op.bTileMap(), {tile[2], tile[1]});
    auto c = impl.prepareOperand(op.c(), op.cAccessMap(), op.getOperandsForC(),
                                 op.cTileMap(), {tile[0], tile[1]});
    SmallVector<Value, 3> argsa{a.leadingDimStride,
                                b.leadingDimStride, c.leadingDimStride};
    SmallVector<Value, 6> argsb{a.memref,           b.memref,
                                c.memref           };
    for (auto i : impl.tile) {
      argsa.push_back(impl.createConstantIntOp(i));
    }

    auto xsmmdis = rewriter.create<CallOp>(op.getLoc(), symbola, rewriter.getIntegerType(64), argsa );
    argsb.push_back(xsmmdis.getResult(0)); 
    rewriter.create<CallOp>(op.getLoc(), symbolb, ArrayRef<Type>{}, argsb );
    rewriter.replaceOp(op, op.c());
    return success();
  }

  struct Impl {
    xsmm::GemmOp op;
    PatternRewriter &rewriter;
    Location loc;
    ModuleOp module;
    Type i32Type;
    Type i64Type;
    Type elementType;
    UnrankedMemRefType unrankedType;
    SmallVector<unsigned, 3> tile;

    struct PreparedOperand {
      Value memref;
      Value leadingDimStride;
    };

    Impl(xsmm::GemmOp op, PatternRewriter &rewriter)
        : op(op), rewriter(rewriter), loc(op.getLoc()),
          module(op.getParentOfType<ModuleOp>()),
          i32Type(rewriter.getIntegerType(32)),
          i64Type(rewriter.getIntegerType(64)),
          elementType(rewriter.getF32Type()),
          unrankedType(
              UnrankedMemRefType::get(elementType, /*memorySpace=*/0)) {
      for (auto attr : op.tile().getValue()) {
        tile.push_back(attr.cast<IntegerAttr>().getInt());
      }
    }

    FlatSymbolRefAttr getOrInsertFunc() {
      const char *symbol = "plaidml_rt_xsmm_gemm_f32";
      auto context = module.getContext();
      if (module.lookupSymbol(symbol)) {
        return SymbolRefAttr::get(symbol, context);
      }
      OpBuilder builder(module.getBodyRegion());
      std::array<Type, 9> inputs{unrankedType, unrankedType, unrankedType,
                                 i32Type,      i32Type,      i32Type,
                                 i32Type,      i32Type,      i32Type};
      ArrayRef<Type> results{};
      auto funcType = builder.getFunctionType(inputs, results);
      ArrayRef<NamedAttribute> attrs{};
      builder.create<FuncOp>(loc, symbol, funcType, attrs);
      return SymbolRefAttr::get(symbol, context);
    }

    FlatSymbolRefAttr getOrInsertDispatchFunc() {
      const char *symbol = "plaidml_rt_xsmm_dispatch_gemm_f32";
      auto context = module.getContext();
      if (module.lookupSymbol(symbol)) {
        return SymbolRefAttr::get(symbol, context);
      }
      OpBuilder builder(module.getBodyRegion());
      std::array<Type, 6> inputs{i32Type,      i32Type,      i32Type,
                                 i32Type,      i32Type,      i32Type};
      auto funcType = builder.getFunctionType(inputs, i64Type);
      ArrayRef<NamedAttribute> attrs{};
      builder.create<FuncOp>(loc, symbol, funcType, attrs);
      return SymbolRefAttr::get(symbol, context);
    }

    FlatSymbolRefAttr getOrInsertExecFunc() {
      const char *symbol = "plaidml_rt_xsmm_exec_gemm_f32";
      auto context = module.getContext();
      if (module.lookupSymbol(symbol)) {
        return SymbolRefAttr::get(symbol, context);
      }
      OpBuilder builder(module.getBodyRegion());
      std::array<Type, 4> inputs{unrankedType, unrankedType, unrankedType, i64Type};
      ArrayRef<Type> results{};
      auto funcType = builder.getFunctionType(inputs, results);
      ArrayRef<NamedAttribute> attrs{};
      builder.create<FuncOp>(loc, symbol, funcType, attrs);
      return SymbolRefAttr::get(symbol, context);
    }

    PreparedOperand prepareOperand(Value operand, AffineMap accessMap,
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
      auto cast = rewriter.create<MemRefCastOp>(loc, subview, unrankedType);

      auto layoutMap = makeStridedLinearLayoutMap(outerStrides, outerOffset,
                                                  module.getContext());
      auto stridesArray = computeStrideArray(layoutMap.compose(tileMap));
      assert(stridesArray.hasValue() && "computeStrideArray must succeed");

      int64_t leadingDimStride = stridesArray->strides[0];
      auto leadingDimValue = createConstantIntOp(leadingDimStride);
      return {cast, leadingDimValue};
    }

    Value createConstantIntOp(int64_t value) {
      return rewriter.create<ConstantIntOp>(loc, value, i32Type);
    }
  };
};

class LowerXSMMPass : public XSMMLoweringBase<LowerXSMMPass> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    patterns.insert<XSMMGemmLowering>(&getContext());
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
