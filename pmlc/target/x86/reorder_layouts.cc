// Copyright 2021, Intel Corporation

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "pmlc/target/x86/pass_detail.h"
#include "pmlc/target/x86/passes.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/matchers.h"

using namespace mlir; // NOLINT
using mlir::matchers::m_Val;

namespace pmlc::target::x86 {

namespace {

struct ConvOperand {
  Value value;
  RankedTensorType type;
  AffineMap idxMap;

  ConvOperand(ValueRange values, ArrayRef<AffineMap> idxMaps, int64_t i)
      : value(values[i]), type(value.getType().cast<RankedTensorType>()),
        idxMap(idxMaps[i]) {}
};

struct ConvCapture {
  ConvOperand input;
  ConvOperand filter;
  ConvOperand output;

  ConvCapture(ValueRange values, ArrayRef<AffineMap> idxMaps,
              ArrayRef<int64_t> order)
      : input(values, idxMaps, order[0]), filter(values, idxMaps, order[1]),
        output(values, idxMaps, order[2]) {}

  RankedTensorType getBlockedInputType(int64_t blockSize) {
    ArrayRef<int64_t> shape = input.type.getShape();
    return RankedTensorType::get({shape[0],             //
                                  shape[3] / blockSize, //
                                  shape[1],             //
                                  shape[2],             //
                                  blockSize},
                                 input.type.getElementType());
  }

  RankedTensorType getBlockedFilterType(int64_t blockSize) {
    ArrayRef<int64_t> shape = filter.type.getShape();
    return RankedTensorType::get({shape[2] / blockSize, //
                                  shape[3] / blockSize, //
                                  shape[0],             //
                                  shape[1],             //
                                  blockSize,            //
                                  blockSize},
                                 filter.type.getElementType());
  }

  RankedTensorType getBlockedOutputType(int64_t blockSize) {
    ArrayRef<int64_t> shape = output.type.getShape();
    return RankedTensorType::get({shape[0],             //
                                  shape[3] / blockSize, //
                                  shape[1],             //
                                  shape[2],             //
                                  blockSize},
                                 input.type.getElementType());
  }
};

static Optional<ConvCapture> detectConv(linalg::GenericOp op) {
  if (op.getNumInputs() != 2 || op.getNumOutputs() != 1)
    return None;

  Block *block = op.getBody();
  Block::BlockArgListType args = block->getArguments();
  if (args.size() != 3)
    return None;

  ValueRange values = op.getOperands();
  SmallVector<AffineMap> idxMaps = op.getIndexingMaps();

  Operation *yieldOp = block->getTerminator();
  if (matchPattern(
          yieldOp,
          m_Op<linalg::YieldOp>(m_Op<AddFOp>(
              m_Val(args[2]), m_Op<MulFOp>(m_Val(args[0]), m_Val(args[1]))))) ||
      matchPattern(yieldOp,
                   m_Op<linalg::YieldOp>(m_Op<AddFOp>(m_Op<MulFOp>(
                                             m_Val(args[0]), m_Val(args[1]))),
                                         m_Val(args[2]))))
    return ConvCapture{values, idxMaps, {0, 1, 2}};

  return None;
}

static AffineExpr getBlockedExpr(MLIRContext *context, int64_t hiDim,
                                 int64_t loDim, int64_t blockSize) {
  return getAffineDimExpr(hiDim, context) * blockSize +
         getAffineDimExpr(loDim, context);
}

// Forward a reorder thru simple elementwise operations.
struct PropagateReorderThruEltwiseOpPattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  struct ReorderInfo {
    linalg::GenericOp reorderOp;
    Value sourceValue;           // The value to be forwarded.
    RankedTensorType sourceType; // The type to be forwarded.
    AffineMap sourceMap;         // The indexingMap tied to the input operand.
    AffineMap sinkMap;           // The indexingMap tied to the output operand.
  };

  static Optional<ReorderInfo> getReorderInfo(linalg::GenericOp op,
                                              OpOperand *operand) {
    if (linalg::GenericOp reorderOp = dyn_cast_or_null<linalg::GenericOp>(
            operand->get().getDefiningOp())) {
      if (reorderOp->hasAttr("reorder")) {
        OpOperand *source = reorderOp.getInputOperand(0);
        AffineMap sourceMap = reorderOp.getTiedIndexingMap(source);
        OpOperand *sink = reorderOp.getOutputOperand(0);
        AffineMap sinkMap = reorderOp.getTiedIndexingMap(sink);
        Value sourceValue = source->get();
        auto sourceType = sourceValue.getType().cast<RankedTensorType>();
        return ReorderInfo{reorderOp, sourceValue, sourceType, sourceMap,
                           sinkMap};
      }
    }
    return None;
  }

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const final {
    // Restrict this pattern to only simple elementwise operations.
    if (op->hasAttr("reorder") ||
        op.getNumParallelLoops() != op.getNumLoops() ||
        op.getNumOutputs() != 1 || op.isInitTensor(op.getOutputOperand(0)))
      return failure();

    // Phase 1: Compute the primary reorder.
    // Computing a primary reorder is here to handle the case where more than
    // one operand is defined by a previous reorder. We only handle the case
    // where a single unique sourceMap and sourceType can be found.
    Optional<ReorderInfo> primary;
    DenseMap<OpOperand *, ReorderInfo> operandInfos;
    for (OpOperand *operand : op.getInputOperands()) {
      // Check that all accesses are 'simple'.
      AffineMap accessMap = op.getTiedIndexingMap(operand);
      if (!accessMap.isProjectedPermutation())
        return failure();

      Optional<ReorderInfo> info = getReorderInfo(op, operand);
      if (info) {
        if (primary) {
          if (info->sourceMap != primary->sourceMap ||
              info->sourceType != primary->sourceType)
            return failure();
        } else {
          primary = info;
        }
        operandInfos[operand] = *info;
      }
    }

    // There's nothing to do if none of the operands are defined by a previous
    // reorder.
    if (!primary)
      return failure();

    // IVLOG(1, "candidate: " << debugString(op));

    // Phase 2: Collect values and indexingMaps.
    SmallVector<Value, 3> inputs;
    SmallVector<AffineMap, 3> indexingMaps;
    for (OpOperand *operand : op.getInputOperands()) {
      auto it = operandInfos.find(operand);
      if (it == operandInfos.end()) {
        AffineMap accessMap = op.getTiedIndexingMap(operand);
        inputs.push_back(operand->get());
        // A composition is used to handle broadcasts.
        // For example, in a bias add operation, the original (unpacked)
        // indexingMap for the bias tensor might be:
        //   (n, h, w, c) -> (c)
        // The primary sinkMap might be:
        //   (n, c0, h, w, c1) -> (n, h, w, c0 * 16 + c1)
        // Thus the composition will be:
        //   (n, c0, h, w, c1) -> (c0 * 16 + c1)
        indexingMaps.push_back(accessMap.compose(primary->sinkMap));
      } else {
        inputs.push_back(it->second.sourceValue);
        indexingMaps.push_back(it->second.sourceMap);
      }
    }
    // The output indexingMap will match the primary input indexingMap.
    indexingMaps.push_back(primary->sourceMap);

    auto init = rewriter.create<linalg::InitTensorOp>(
        op->getLoc(), primary->sourceType.getShape(),
        primary->sourceType.getElementType());

    SmallVector<StringRef> iterTypes(primary->sourceType.getRank(), "parallel");
    auto newOp = rewriter.create<linalg::GenericOp>(
        op->getLoc(),
        /*resultTensorTypes=*/TypeRange{primary->sourceType},
        /*inputs=*/inputs,
        /*outputs=*/ValueRange{init},
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iterTypes);
    rewriter.cloneRegionBefore(op.region(), newOp.region(),
                               newOp.region().end());

    BlockAndValueMapping mapper;
    mapper.map(primary->sourceValue, newOp.getResult(0));
    auto newReorderOp = cast<linalg::GenericOp>(rewriter.cloneWithoutRegions(
        *primary->reorderOp.getOperation(), mapper));
    rewriter.inlineRegionBefore(primary->reorderOp.region(),
                                newReorderOp.region(),
                                newReorderOp.region().begin());

    rewriter.replaceOp(op, newReorderOp.getResults());

    return success();
  }
};

struct ReorderLayoutsPass : public ReorderLayoutsBase<ReorderLayoutsPass> {
  void runOnFunction() final {
    FuncOp func = getFunction();
    MLIRContext *context = func.getContext();

    func.walk([&](linalg::GenericOp op) { reorderConvolution(op); });

    RewritePatternSet patterns(context);
    patterns.add<PropagateReorderThruEltwiseOpPattern>(context);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }

  Value reorderConvolution(linalg::GenericOp op) {
    constexpr int64_t blockSize = 16;

    Optional<ConvCapture> conv = detectConv(op);
    if (!conv)
      return nullptr;

    // IVLOG(1, "Conv: " << debugString(op));
    Value initValue;
    if (!matchPattern(op.getOutputOperand(0)->get(),
                      m_Op<linalg::FillOp>(m_Capture(&initValue),
                                           m_Op<linalg::InitTensorOp>()))) {
      op.emitWarning("GenericOp uses an unrecognized init pattern.");
      return nullptr;
    }

    // dims: (n, h, w, c, r, s, k)
    Optional<SmallVector<int64_t, 4>> ranges = op.getStaticLoopRanges();
    if (!ranges) {
      op.emitWarning("GenericOp does not have static ranges.");
      return nullptr;
    }

    if (ranges->size() != 7 ||           //
        (*ranges)[3] % blockSize != 0 || // C
        (*ranges)[6] % blockSize != 0)   // K
      return nullptr;

    MLIRContext *context = &getContext();
    ImplicitLocOpBuilder builder(op->getLoc(), op);

    // Reorder input
    RankedTensorType blockedInputType = conv->getBlockedInputType(blockSize);

    // (n, c0, h, w, c1) -> (n, h, w, c0 * 16 + c1)
    AffineMap inputSourceMap =
        AffineMap::get(5, 0,
                       ArrayRef<AffineExpr>{
                           getAffineDimExpr(0, context),
                           getAffineDimExpr(2, context),
                           getAffineDimExpr(3, context),
                           getBlockedExpr(context, 1, 4, blockSize),
                       },
                       context);

    // (n, c0, h, w, c1) -> (n, c0, h, w, c1)
    AffineMap inputSinkMap = AffineMap::getMultiDimIdentityMap(5, context);

    linalg::GenericOp reorderInput =
        createReorderLoop(builder, blockedInputType, conv->input.value,
                          inputSourceMap, inputSinkMap);

    // Reorder filter
    RankedTensorType blockedFilterType = conv->getBlockedFilterType(blockSize);

    // (k1, c1, r, s, k0, c0) -> (r, s, k1 * 16 + k0, c1 * 16 + c0)
    AffineMap filterSourceMap =
        AffineMap::get(6, 0,
                       ArrayRef<AffineExpr>{
                           getAffineDimExpr(2, context),
                           getAffineDimExpr(3, context),
                           getBlockedExpr(context, 0, 4, blockSize),
                           getBlockedExpr(context, 1, 5, blockSize),
                       },
                       context);

    // (k1, c1, r, s, k0, c0) -> (k1, c1, r, s, k0, c0)
    AffineMap filterSinkMap = AffineMap::getMultiDimIdentityMap(6, context);

    linalg::GenericOp reorderFilter =
        createReorderLoop(builder, blockedFilterType, conv->filter.value,
                          filterSourceMap, filterSinkMap);

    // Adjust convolution
    RankedTensorType blockedOutputType = conv->getBlockedOutputType(blockSize);

    // oldInput = (n, h, w, c0, r, s, k0) -> (n, h + r, w + s, k0)
    // newInput = (n, h, w, c0, r, s, k0, c1, k1) -> (n, k1, h + r, w + s, k0)
    AffineMap newInputMap = AffineMap::get(9, 0,
                                           ArrayRef<AffineExpr>{
                                               conv->input.idxMap.getResult(0),
                                               getAffineDimExpr(8, context),
                                               conv->input.idxMap.getResult(1),
                                               conv->input.idxMap.getResult(2),
                                               conv->input.idxMap.getResult(3),
                                           },
                                           context);

    // oldFilter = (n, h, w, c, r, s, k) -> (r, s, k, c)
    // newFilter = (n, h, w, c0, r, s, k0, c1, k1) -> (k1, c1, r, s, k0, c0)
    AffineMap newFilterMap =
        AffineMap::get(9, 0,
                       ArrayRef<AffineExpr>{
                           getAffineDimExpr(8, context),
                           getAffineDimExpr(7, context),
                           conv->filter.idxMap.getResult(0),
                           conv->filter.idxMap.getResult(1),
                           conv->filter.idxMap.getResult(2),
                           conv->filter.idxMap.getResult(3),
                       },
                       context);

    // oldOutput = (n, h, w, c, r, s, k) -> (n, h, w, c)
    // newOutput = (n, h, w, c0, r, s, k0, c1, k1) -> (n, c1, h, w, c0)
    AffineMap newOutputMap =
        AffineMap::get(9, 0,
                       ArrayRef<AffineExpr>{
                           conv->output.idxMap.getResult(0),
                           getAffineDimExpr(7, context),
                           conv->output.idxMap.getResult(1),
                           conv->output.idxMap.getResult(2),
                           conv->output.idxMap.getResult(3),
                       },
                       context);

    auto init = builder.create<linalg::InitTensorOp>(
        blockedOutputType.getShape(), blockedOutputType.getElementType());
    auto fill = builder.create<linalg::FillOp>(initValue, init);
    auto newConv = builder.create<linalg::GenericOp>(
        TypeRange{blockedOutputType},
        ValueRange{reorderInput.getResult(0), reorderFilter.getResult(0)},
        ValueRange{fill.result()},
        ArrayRef<AffineMap>{newInputMap, newFilterMap, newOutputMap},
        ArrayRef<StringRef>{
            "parallel",  // N
            "parallel",  // H
            "parallel",  // W
            "parallel",  // C0
            "reduction", // R
            "reduction", // S
            "reduction", // K0
            "parallel",  // C1
            "reduction", // K1
        },
        /*doc=*/"",
        /*libraryCall=*/"",
        [](OpBuilder &builder, Location loc, ValueRange args) {
          auto mul = builder.create<MulFOp>(loc, args[0], args[1]);
          auto add = builder.create<AddFOp>(loc, args[2], mul);
          builder.create<linalg::YieldOp>(loc, ValueRange{add});
        });

    // Reorder output
    linalg::GenericOp reorderOutput =
        createReorderLoop(builder, conv->output.type, newConv.getResult(0),
                          inputSinkMap, inputSourceMap);

    op.getResult(0).replaceAllUsesWith(reorderOutput.getResult(0));
    op.erase();

    reorderOutput->setAttr("reorder", builder.getUnitAttr());

    return reorderOutput.getResult(0);
  }

  linalg::GenericOp createReorderLoop(ImplicitLocOpBuilder &builder,
                                      RankedTensorType resultType, Value source,
                                      AffineMap sourceMap, AffineMap sinkMap) {
    unsigned numDims = sourceMap.getNumDims();
    auto init = builder.create<linalg::InitTensorOp>(
        resultType.getShape(), resultType.getElementType());

    SmallVector<StringRef> iterTypes(numDims, "parallel");
    return builder.create<linalg::GenericOp>(
        /*resultTensorTypes=*/TypeRange{resultType},
        /*inputs=*/ValueRange{source},
        /*outputs=*/ValueRange{init},
        /*indexingMaps=*/ArrayRef<AffineMap>{sourceMap, sinkMap},
        /*iteratorTypes=*/iterTypes,
        /*doc=*/"",
        /*libraryCall=*/"",
        [](OpBuilder &builder, Location loc, ValueRange args) {
          builder.create<linalg::YieldOp>(loc, args[0]);
        });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createReorderLayoutsPass() {
  return std::make_unique<ReorderLayoutsPass>();
}

} // namespace pmlc::target::x86
