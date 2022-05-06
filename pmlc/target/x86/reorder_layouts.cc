// Copyright 2021, Intel Corporation

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "pmlc/dialect/linalgx/analysis/convolution.h"
#include "pmlc/target/x86/pass_detail.h"
#include "pmlc/target/x86/passes.h"
#include "pmlc/util/env.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/matchers.h"
#include "pmlc/util/util.h"

using namespace mlir; // NOLINT
using mlir::matchers::m_Val;

namespace pmlc::target::x86 {

namespace linalgx = dialect::linalgx;

namespace {

static AffineExpr getBlockedExpr(MLIRContext *context, int64_t highDim,
                                 int64_t lowDim, int64_t blockSize) {
  return getAffineDimExpr(highDim, context) * blockSize +
         getAffineDimExpr(lowDim, context);
}

struct ReorderInfo {
  linalgx::CopyOp reorderOp;
  Value sourceValue;           // The value to be forwarded.
  RankedTensorType sourceType; // The type to be forwarded.
  AffineMap sourceMap;         // The indexingMap tied to the input operand.
  AffineMap sinkMap;           // The indexingMap tied to the output operand.
};

static Optional<ReorderInfo> getReorderInfo(OpOperand *operand) {
  if (auto reorderOp =
          dyn_cast_or_null<linalgx::CopyOp>(operand->get().getDefiningOp())) {
    OpOperand *source = reorderOp.getInputOperand(0);
    AffineMap sourceMap = reorderOp.getTiedIndexingMap(source);
    OpOperand *sink = reorderOp.getOutputOperand(0);
    AffineMap sinkMap = reorderOp.getTiedIndexingMap(sink);
    Value sourceValue = source->get();
    auto sourceType = sourceValue.getType().cast<RankedTensorType>();
    return ReorderInfo{reorderOp, sourceValue, sourceType, sourceMap, sinkMap};
  }
  return None;
}

static bool hasDynamicPad(ArrayRef<int64_t> array) {
  return llvm::any_of(
      array, [](int64_t value) { return ShapedType::isDynamic(value); });
}

static SmallVector<int64_t, 4> extractVector(ArrayAttr arrayAttr) {
  return llvm::to_vector<4>(
      llvm::map_range(arrayAttr.getAsRange<IntegerAttr>(),
                      [](IntegerAttr attr) { return attr.getInt(); }));
}

// check that we have (channels-last logical ordering):
// (n, h, w, c), (r, s, c, k) -> (n, h, w, k) or
// (n, h, w, c), (r, s, c, 1) -> (n, h, w, c)
bool testChannels(AffineExpr cin, AffineExpr cout, AffineExpr fin,
                  AffineExpr fout) {
  if (cin != fin) {
    return false;
  }
  if (cout == fout) {
    return true;
  }
  if (cin == cout && fout == getAffineConstantExpr(1, fout.getContext())) {
    return true;
  }
  return false;
}

// Return AffineConstantExpr(1) if expr's range is 1.
// Otherwise, return expr directly.
AffineExpr isSizeOne(AffineExpr expr, SmallVector<int64_t, 4> &ranges) {
  if (auto dim = expr.dyn_cast<AffineDimExpr>()) {
    if (ranges[dim.getPosition()] == 1) {
      return getAffineConstantExpr(1, expr.getContext());
    }
  }
  return expr;
}

// Forward a reorder thru linalg.pad_tensor operations.
struct PropagateReorderThruPadTensorOpPattern
    : public OpRewritePattern<linalg::PadTensorOp> {
  using OpRewritePattern<linalg::PadTensorOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::PadTensorOp op,
                                PatternRewriter &rewriter) const final {
    Optional<ReorderInfo> info = getReorderInfo(&op->getOpOperand(0));
    if (!info) {
      IVLOG(1, "Unrecognized PadTensorOp: " << debugString(op));
      return failure();
    }

    SmallVector<int64_t, 4> lower = extractVector(op.static_low());
    SmallVector<int64_t, 4> upper = extractVector(op.static_high());

    if (hasDynamicPad(lower) || hasDynamicPad(upper)) {
      return failure();
    }

    lower.insert(lower.begin() + 1, 0);
    upper.insert(upper.begin() + 1, 0);

    auto newOp =
        rewriter.create<linalg::PadTensorOp>(op->getLoc(),
                                             /*source=*/info->sourceValue,
                                             /*staticLow=*/lower,
                                             /*staticHigh=*/upper,
                                             /*low=*/ValueRange{},
                                             /*high=*/ValueRange{});
    SmallVector<Type, 4> padArgs(lower.size(), rewriter.getIndexType());
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.createBlock(&newOp.region(), newOp.region().begin(), padArgs);
      rewriter.create<linalg::YieldOp>(
          op->getLoc(), ValueRange{op.getConstantPaddingValue()});
    }

    RankedTensorType resultType = op.getResultType();
    linalgx::CopyOp newReorderOp = rewriter.create<linalgx::CopyOp>(
        op->getLoc(), newOp.getResult(),
        rewriter.create<linalg::InitTensorOp>(
            op->getLoc(), resultType.getShape(), resultType.getElementType()),
        info->sourceMap, info->sinkMap);

    rewriter.replaceOp(op, newReorderOp.getResult());

    return success();
  }
};

// Forward a reorder thru simple elementwise operations.
struct PropagateReorderThruEltwiseOpPattern
    : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const final {
    // Restrict this pattern to only simple elementwise operations.
    if (op.getNumParallelLoops() != op.getNumLoops() ||
        op.getNumOutputs() != 1 || op.isInitTensor(op.getOutputOperand(0)))
      return failure();

    OpOperand *initOperand = op.getOutputOperand(0);
    AffineMap initMap = op.getTiedIndexingMap(initOperand);
    if (!initMap.isMinorIdentity()) {
      IVLOG(1, "Cannot propagate reorder thru op with complex result: "
                   << debugString(op));
      return failure();
    }

    auto initType = initOperand->get().getType().dyn_cast<RankedTensorType>();
    if (!initType || !initType.getRank())
      return failure();
    int64_t channels = initType.getShape().back();

    // Phase 1: Compute the primary reorder.
    // Computing a primary reorder is here to handle the case where more than
    // one operand is defined by a previous reorder. We only handle the case
    // where a single shared sourceMap and sourceType can be found.
    Optional<ReorderInfo> primary;
    DenseMap<OpOperand *, ReorderInfo> operandInfos;
    for (OpOperand *operand : op.getInputOperands()) {
      // Check that all accesses are 'simple'.
      AffineMap accessMap = op.getTiedIndexingMap(operand);
      if (!accessMap.isProjectedPermutation()) {
        IVLOG(1, "Cannot propagate reorder thru op with complex operand: "
                     << debugString(op));
        return failure();
      }

      if (auto operandType =
              operand->get().getType().dyn_cast<RankedTensorType>()) {
        if (!operandType.getRank() ||
            operandType.getShape().back() != channels) {
          IVLOG(1, "Cannot propagate reorder thru op with mismatched channels: "
                       << debugString(op));
          return failure();
        }
      }

      if (Optional<ReorderInfo> info = getReorderInfo(operand)) {
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

    IVLOG(3, "candidate: " << debugString(op));

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
        //   (n, c0, h, w, c1) -> (n, h, w, c0 * B + c1)
        // Thus the composition will be:
        //   (n, c0, h, w, c1) -> (c0 * B + c1)
        indexingMaps.push_back(accessMap.compose(primary->sinkMap));
      } else {
        inputs.push_back(it->second.sourceValue);
        indexingMaps.push_back(it->second.sourceMap);
      }
    }
    // The output indexingMap will match the primary input indexingMap.
    indexingMaps.push_back(primary->sourceMap);

    auto resultType = op.getResult(0).getType().cast<RankedTensorType>();
    auto init = rewriter.create<linalg::InitTensorOp>(
        op->getLoc(), primary->sourceType.getShape(),
        resultType.getElementType());

    RankedTensorType newResultType = RankedTensorType::get(
        primary->sourceType.getShape(), resultType.getElementType());

    SmallVector<StringRef> iterTypes(primary->sourceType.getRank(),
                                     getParallelIteratorTypeName());
    auto newOp = rewriter.create<linalg::GenericOp>(
        op->getLoc(),
        /*resultTensorTypes=*/TypeRange{newResultType},
        /*inputs=*/inputs,
        /*outputs=*/ValueRange{init},
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iterTypes);
    rewriter.cloneRegionBefore(op.region(), newOp.region(),
                               newOp.region().end());

    linalgx::CopyOp newReorderOp = rewriter.create<linalgx::CopyOp>(
        op->getLoc(), newOp.getResult(0),
        rewriter.create<linalg::InitTensorOp>(
            op->getLoc(), resultType.getShape(), resultType.getElementType()),
        primary->sourceMap, primary->sinkMap);

    rewriter.replaceOp(op, newReorderOp.getResult());

    return success();
  }
};

struct FoldReordersPattern : public OpRewritePattern<linalgx::CopyOp> {
  using OpRewritePattern<linalgx::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalgx::CopyOp op,
                                PatternRewriter &rewriter) const final {
    OpOperand *input = op.getInputOperand(0);
    linalgx::CopyOp predecessor =
        dyn_cast_or_null<linalgx::CopyOp>(input->get().getDefiningOp());
    if (!predecessor)
      return failure();

    OpOperand *initialInput = predecessor.getInputOperand(0);
    Type initialType = initialInput->get().getType();
    AffineMap initialMap = predecessor.getTiedIndexingMap(initialInput);

    OpOperand *finalOutput = op.getOutputOperand(0);
    Type finalType = finalOutput->get().getType();
    AffineMap finalMap = op.getTiedIndexingMap(finalOutput);

    if (initialType != finalType || initialMap != finalMap)
      return failure();

    op.getResult().replaceAllUsesWith(initialInput->get());

    return success();
  }
};

struct FoldBroadcastReordersPattern : public OpRewritePattern<linalgx::CopyOp> {
  using OpRewritePattern<linalgx::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalgx::CopyOp op,
                                PatternRewriter &rewriter) const final {
    OpOperand *input = op.getInputOperand(0);
    linalg::GenericOp predecessor =
        dyn_cast_or_null<linalg::GenericOp>(input->get().getDefiningOp());
    if (!predecessor)
      return failure();

    if (predecessor.getNumInputs() != 1 || predecessor.getNumOutputs() != 1 ||
        predecessor.getNumParallelLoops() != predecessor.getNumLoops())
      return failure();

    MLIRContext *context = getContext();

    OpOperand *initialInput = predecessor.getInputOperand(0);
    AffineMap initialInputMap = predecessor.getTiedIndexingMap(initialInput);
    AffineMap initialInputPattern =
        AffineMap::get(4, 0,
                       ArrayRef<AffineExpr>{
                           getAffineDimExpr(3, context),
                       },
                       context);
    if (initialInputMap != initialInputPattern)
      return failure();

    OpOperand *initialOutput = predecessor.getOutputOperand(0);
    AffineMap initialOutputMap = predecessor.getTiedIndexingMap(initialOutput);
    if (!initialOutputMap.isIdentity())
      return failure();

    Block *block = predecessor.getBody();
    Block::BlockArgListType args = block->getArguments();
    if (args.size() != 2)
      return failure();

    Operation *yieldOp = block->getTerminator();
    if (!matchPattern(yieldOp, m_Op<linalg::YieldOp>(m_Val(args[0]))))
      return failure();

    AffineMap newInputMap = initialInputMap.compose(*op.inputMap());
    SmallVector<AffineMap, 2> indexingMaps = {newInputMap, *op.outputMap()};

    SmallVector<StringRef, 4> iterTypes(newInputMap.getNumDims(),
                                        getParallelIteratorTypeName());
    auto newOp = rewriter.create<linalg::GenericOp>(
        op->getLoc(),
        /*resultTensorTypes=*/op->getResultTypes(),
        /*inputs=*/predecessor.inputs(),
        /*outputs=*/op.outputs(),
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iterTypes);
    rewriter.cloneRegionBefore(op.region(), newOp.region(),
                               newOp.region().end());

    rewriter.replaceOp(op, newOp.getResults());

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
    patterns.add<PropagateReorderThruPadTensorOpPattern>(context);
    patterns.add<FoldReordersPattern>(context);
    patterns.add<FoldBroadcastReordersPattern>(context);
    (void)applyPatternsAndFoldGreedily(
        func, std::move(patterns),
        GreedyRewriteConfig{/*useTopDownTraversal=*/true});
  }

  void reorderConvolution(linalg::GenericOp op) {
    Optional<linalgx::ConvCapture> conv = linalgx::detectConv(op);
    if (!conv) {
      IVLOG(3, "Cannot reorder: not a convolution. " << debugString(op));
      return;
    }

    // check that this is a 2D convolution
    if (conv->input.idxMap.getNumResults() != 4 ||
        conv->filter.idxMap.getNumResults() != 4 ||
        conv->output.idxMap.getNumResults() != 4) {
      IVLOG(1,
            "Cannot reorder: expected 2D convolution. op: " << debugString(op));
      return;
    }

    // check that we have (channels-last logical ordering):
    // (n, h, w, c), (r, s, c, k) -> (n, h, w, k)
    if (conv->input.idxMap.getResult(3) != conv->filter.idxMap.getResult(2) ||
        conv->filter.idxMap.getResult(3) != conv->output.idxMap.getResult(3)) {
      IVLOG(1, "Cannot reorder: expected channels-last logical ordering.");
      return;
    }

    // dims: (n, h, w, c, r, s, k)
    Optional<SmallVector<int64_t, 4>> ranges = op.getStaticLoopRanges();
    if (!ranges) {
      IVLOG(1, "Cannot reorder: expected static ranges.");
      return;
    }

    if (ranges->size() != 7) {
      IVLOG(1, "Cannot reorder: number of indexes is not 7.");
      return;
    }

    SmallVector<int64_t, 4> feasibleSizes = {32, 16};
    int64_t blockSize = 0;
    std::string blockSizeStr = util::getEnvVar("PLAIDML_BLOCK_SIZE");
    if (!blockSizeStr.empty()) {
      blockSize = std::stoi(blockSizeStr);
    }
    for (int64_t size : feasibleSizes) {
      if ((*ranges)[3] % size == 0 && (*ranges)[6] % size == 0) {
        blockSize = size;
        break;
      }
    }

    if (blockSize == 0) {
      IVLOG(1, "Cannot reorder: incompatible layout. op: " << debugString(op));
      return;
    }

    MLIRContext *context = &getContext();
    ImplicitLocOpBuilder builder(op->getLoc(), op);

    // Reorder input
    RankedTensorType blockedInputType = conv->getBlockedInputType(blockSize);

    // (n, c1, h, w, c0) -> (n, h, w, c1 * B + c0)
    AffineMap inputSourceMap =
        AffineMap::get(5, 0,
                       ArrayRef<AffineExpr>{
                           getAffineDimExpr(0, context),
                           getAffineDimExpr(2, context),
                           getAffineDimExpr(3, context),
                           getBlockedExpr(context, 1, 4, blockSize),
                       },
                       context);

    // (n, c1, h, w, c0) -> (n, c1, h, w, c0)
    AffineMap inputSinkMap = AffineMap::getMultiDimIdentityMap(5, context);

    linalgx::CopyOp reorderInput =
        createReorderOp(builder, blockedInputType, conv->input.value,
                        inputSourceMap, inputSinkMap);

    // Reorder filter
    RankedTensorType blockedFilterType = conv->getBlockedFilterType(blockSize);

    // (k1, c1, r, s, k0, c0) -> (r, s, k1 * B + k0, c1 * B + c0)
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

    linalgx::CopyOp reorderFilter =
        createReorderOp(builder, blockedFilterType, conv->filter.value,
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

    linalgx::CopyOp reorderInit =
        createReorderOp(builder, blockedOutputType, conv->output.value,
                        inputSourceMap, inputSinkMap);

    auto newConv = builder.create<linalg::GenericOp>(
        TypeRange{blockedOutputType},
        ValueRange{reorderInput.getResult(), reorderFilter.getResult()},
        ValueRange{reorderInit.getResult()},
        ArrayRef<AffineMap>{newInputMap, newFilterMap, newOutputMap},
        ArrayRef<StringRef>{
            getParallelIteratorTypeName(),  // N
            getParallelIteratorTypeName(),  // H
            getParallelIteratorTypeName(),  // W
            getParallelIteratorTypeName(),  // C0
            getReductionIteratorTypeName(), // R
            getReductionIteratorTypeName(), // S
            getReductionIteratorTypeName(), // K0
            getParallelIteratorTypeName(),  // C1
            getReductionIteratorTypeName(), // K1
        },
        /*doc=*/"",
        /*libraryCall=*/"",
        [](OpBuilder &builder, Location loc, ValueRange args) {
          auto mul = builder.create<MulFOp>(loc, args[0], args[1]);
          auto add = builder.create<AddFOp>(loc, args[2], mul);
          builder.create<linalg::YieldOp>(loc, ValueRange{add});
        });

    // Reorder output
    linalgx::CopyOp reorderOutput =
        createReorderOp(builder, conv->output.type, newConv.getResult(0),
                        inputSinkMap, inputSourceMap);
    op.getResult(0).replaceAllUsesWith(reorderOutput.getResult());
    op.erase();
  }

  linalgx::CopyOp createReorderOp(ImplicitLocOpBuilder &builder,
                                  RankedTensorType resultType, Value source,
                                  AffineMap sourceMap, AffineMap sinkMap) {
    auto init = builder.create<linalg::InitTensorOp>(
        resultType.getShape(), resultType.getElementType());
    return builder.create<linalgx::CopyOp>(source, init, sourceMap, sinkMap);
  }
};

struct ReorderWeightLayoutsPass
    : public ReorderWeightLayoutsBase<ReorderWeightLayoutsPass> {
  void runOnFunction() final {
    FuncOp func = getFunction();
    MLIRContext *context = func.getContext();

    func.walk([&](linalg::GenericOp op) { reorderConvolution(op); });
  }

  void reorderConvolution(linalg::GenericOp op) {
    Optional<linalgx::ConvCapture> conv = linalgx::detectConv(op);
    if (!conv)
      return;

    // check that this is a 2D convolution
    if (conv->input.idxMap.getNumResults() != 4 ||
        conv->filter.idxMap.getNumResults() != 4 ||
        conv->output.idxMap.getNumResults() != 4) {
      IVLOG(1, "Cannot reorder: expected 2D convolution.");
      return;
    }

    // dims: (n, h, w, c, r, s, k)
    Optional<SmallVector<int64_t, 4>> ranges = op.getStaticLoopRanges();
    if (!ranges) {
      IVLOG(1, "Cannot reorder: expected static ranges.");
      return;
    }

    if (ranges->size() != 7) {
      IVLOG(1, "Cannot reorder: number of indexes is not 7.");
      return;
    }

    // check that we have (channels-last logical ordering):
    if (!testChannels(conv->input.idxMap.getResult(3),
                      conv->output.idxMap.getResult(3),
                      conv->filter.idxMap.getResult(2),
                      isSizeOne(conv->filter.idxMap.getResult(3), *ranges))) {
      IVLOG(1, "Cannot reorder: expected channels-last logical ordering.");
      return;
    }

    SmallVector<int64_t, 4> feasibleSizes = {32, 16};
    int64_t blockSize = 0;
    std::string blockSizeStr = util::getEnvVar("PLAIDML_BLOCK_SIZE");
    if (!blockSizeStr.empty()) {
      blockSize = std::stoi(blockSizeStr);
    }
    for (int64_t size : feasibleSizes) {
      if ((*ranges)[3] % size == 0 &&
          ((*ranges)[6] % size == 0 || (*ranges)[6] == 1)) {
        blockSize = size;
        break;
      }
    }

    if (blockSize == 0) {
      IVLOG(1, "Cannot reorder: incompatible layout. op: " << debugString(op));
      return;
    }

    bool oneFilterOut = (*ranges)[6] == 1;

    MLIRContext *context = &getContext();
    ImplicitLocOpBuilder builder(op->getLoc(), op);

    // Reorder input
    RankedTensorType blockedInputType = conv->getBlockedInputType(blockSize);

    // Reorder filter
    RankedTensorType blockedFilterType = conv->getBlockedFilterType(blockSize);

    // (k1, c1, r, s, k0, c0) -> (r, s, k1 * B + k0, c1 * B + c0)
    AffineMap filterSourceMap = AffineMap::get(
        6, 0,
        ArrayRef<AffineExpr>{
            getAffineDimExpr(2, context),
            getAffineDimExpr(3, context),
            getBlockedExpr(context, 0, 4, blockSize),
            oneFilterOut ? getAffineConstantExpr(0, context)
                         : getBlockedExpr(context, 1, 5, blockSize),
        },
        context);

    // (k1, c1, r, s, k0, c0) -> (k1, c1, r, s, k0, c0)
    AffineMap filterSinkMap = AffineMap::getMultiDimIdentityMap(6, context);

    linalgx::CopyOp reorderFilter =
        createReorderOp(builder, blockedFilterType, conv->filter.value,
                        filterSourceMap, filterSinkMap);

    // Adjust convolution
    RankedTensorType blockedOutputType = conv->getBlockedOutputType(blockSize);

    // oldInput = (n, h, w, c0, r, s, k0) -> (n, h + r, w + s, k0)
    // newInput = (n, h, w, c0, r, s, k0, c1, k1) ->
    //            (n, h + r, w + s, k1 * B + k0)
    AffineMap newInputMap =
        AffineMap::get(9, 0,
                       ArrayRef<AffineExpr>{
                           conv->input.idxMap.getResult(0),
                           conv->input.idxMap.getResult(1),
                           conv->input.idxMap.getResult(2),
                           getBlockedExpr(context, 8, 6, blockSize),
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
    // newOutput = (n, h, w, c0, r, s, k0, c1, k1) -> (n, h, w, c1 * B + c0)
    AffineMap newOutputMap =
        AffineMap::get(9, 0,
                       ArrayRef<AffineExpr>{
                           conv->output.idxMap.getResult(0),
                           conv->output.idxMap.getResult(1),
                           conv->output.idxMap.getResult(2),
                           getBlockedExpr(context, 7, 3, blockSize),
                       },
                       context);

    auto newConv = builder.create<linalg::GenericOp>(
        TypeRange{conv->output.type},
        ValueRange{conv->input.value, reorderFilter.getResult()},
        ValueRange{conv->output.value},
        ArrayRef<AffineMap>{newInputMap, newFilterMap, newOutputMap},
        ArrayRef<StringRef>{
            getParallelIteratorTypeName(),  // N
            getParallelIteratorTypeName(),  // H
            getParallelIteratorTypeName(),  // W
            getParallelIteratorTypeName(),  // C0
            getReductionIteratorTypeName(), // R
            getReductionIteratorTypeName(), // S
            getReductionIteratorTypeName(), // K0
            getParallelIteratorTypeName(),  // C1
            getReductionIteratorTypeName(), // K1
        },
        /*doc=*/"",
        /*libraryCall=*/"",
        [](OpBuilder &builder, Location loc, ValueRange args) {
          auto mul = builder.create<MulFOp>(loc, args[0], args[1]);
          auto add = builder.create<AddFOp>(loc, args[2], mul);
          builder.create<linalg::YieldOp>(loc, ValueRange{add});
        });

    op.getResult(0).replaceAllUsesWith(newConv.getResult(0));
    op.erase();
  }

  linalgx::CopyOp createReorderOp(ImplicitLocOpBuilder &builder,
                                  RankedTensorType resultType, Value source,
                                  AffineMap sourceMap, AffineMap sinkMap) {
    auto init = builder.create<linalg::InitTensorOp>(
        resultType.getShape(), resultType.getElementType());
    return builder.create<linalgx::CopyOp>(source, init, sourceMap, sinkMap);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createReorderWeightLayoutsPass() {
  return std::make_unique<ReorderWeightLayoutsPass>();
}

std::unique_ptr<mlir::Pass> createReorderLayoutsPass() {
  return std::make_unique<ReorderLayoutsPass>();
}

} // namespace pmlc::target::x86
