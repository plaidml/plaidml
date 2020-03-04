// Copyright 2020 Intel Corporation

#include "pmlc/target/x86/xsmm_lowering.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/Functional.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/xsmm/ir/dialect.h"
#include "pmlc/dialect/xsmm/ir/ops.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::target::x86 {

namespace xsmm = dialect::xsmm;

namespace {

/// Create a sequence of operations that implement the `affineMap` applied to
/// the given `operands` (as it it were an AffineApplyOp).
static Optional<SmallVector<Value, 8>> expandAffineMap(OpBuilder &builder,
                                                       Location loc,
                                                       AffineMap affineMap,
                                                       ValueRange operands) {
  auto numDims = affineMap.getNumDims();
  auto expanded = functional::map(
      [numDims, &builder, loc, operands](AffineExpr expr) {
        return expandAffineExpr(builder, loc, expr,
                                operands.take_front(numDims),
                                operands.drop_front(numDims));
      },
      affineMap.getResults());
  if (llvm::all_of(expanded, [](Value v) { return v; }))
    return expanded;
  return None;
}

static constexpr int64_t kUnusedDimension = -1;
static constexpr unsigned kDimensionM = 0;
static constexpr unsigned kDimensionN = 1;

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

  PatternMatchResult matchAndRewrite(xsmm::GemmOp op,
                                     PatternRewriter &rewriter) const override {
    Impl impl(op, rewriter);
    auto symbol = impl.getOrInsertFunc();
    auto a = impl.prepareOperand(op.a(), op.aAccessMap(), op.getOperandsForA(),
                                 op.aTileMap(), kDimensionM);
    auto b = impl.prepareOperand(op.b(), op.bAccessMap(), op.getOperandsForB(),
                                 op.bTileMap(), kDimensionN);
    auto c = impl.prepareOperand(op.c(), op.cAccessMap(), op.getOperandsForC(),
                                 op.cTileMap(), kDimensionN);
    SmallVector<Value, 9> args{a.memref,           b.memref,
                               c.memref,           a.leadingDimStride,
                               b.leadingDimStride, c.leadingDimStride};
    for (auto i : impl.tile) {
      args.push_back(impl.createConstantIntOp(i));
    }
    rewriter.replaceOpWithNewOp<CallOp>(op, symbol, ArrayRef<Type>{}, args);
    return matchSuccess();
  }

  struct Impl {
    xsmm::GemmOp op;
    PatternRewriter &rewriter;
    Location loc;
    ModuleOp module;
    Type i32Type;
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

    PreparedOperand prepareOperand(Value operand, AffineMap accessMap,
                                   ValueRange mapOperands, AffineMap tileMap,
                                   unsigned leadingDimPos) {
      ArrayRef<Value> empty{};
      auto offsets = expandAffineMap(rewriter, loc, accessMap, mapOperands);
      auto memRefType = operand.getType().cast<MemRefType>();

      SmallVector<int64_t, 8> shape;
      auto flat = getFlattenedTileDimMapping(tileMap);
      for (auto dim : flat) {
        if (dim == kUnusedDimension)
          shape.push_back(1);
        else
          shape.push_back(tile[dim]);
      }

      int64_t outerOffset;
      SmallVector<int64_t, 4> outerStrides;
      getStridesAndOffset(memRefType, outerStrides, outerOffset);

      auto subviewMap = makeStridedLinearLayoutMap(
          outerStrides, MemRefType::getDynamicStrideOrOffset(),
          module.getContext());

      auto resultType = MemRefType::get(shape, elementType, subviewMap);
      auto subview =
          rewriter.create<SubViewOp>(loc, operand, *offsets, /*sizes=*/empty,
                                     /*strides=*/empty, resultType);
      auto cast = rewriter.create<MemRefCastOp>(loc, subview, unrankedType);

      auto layoutMap = makeStridedLinearLayoutMap(outerStrides, outerOffset,
                                                  module.getContext());
      auto stridesArray = computeStrideArray(layoutMap.compose(tileMap));
      assert(stridesArray.hasValue() && "computeStrideArray must succeed");

      int64_t leadingDimStride = stridesArray->strides[leadingDimPos];
      auto leadingDimValue = createConstantIntOp(leadingDimStride);
      return {cast, leadingDimValue};
    }

    Value createConstantIntOp(int64_t value) {
      return rewriter.create<ConstantIntOp>(loc, value, i32Type);
    }
  };
};

} // namespace

void populateXSMMConversionPatterns(OwningRewritePatternList &patterns,
                                    MLIRContext *ctx) {
  patterns.insert<XSMMGemmLowering>(ctx);
}

namespace {

class LowerXSMMPass : public FunctionPass<LowerXSMMPass> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    populateXSMMConversionPatterns(patterns, &getContext());
    ConversionTarget target(getContext());
    target.addLegalDialect<AffineOpsDialect, StandardOpsDialect>();
    target.addIllegalDialect<xsmm::Dialect>();
    if (failed(applyPartialConversion(getFunction(), target, patterns)))
      signalPassFailure();
  }
};

} // namespace

static PassRegistration<LowerXSMMPass> pass("xsmm",
                                            "XSMM to standard conversion");

} // namespace pmlc::target::x86
