// Copyright 2021, Intel Corporation

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Support/DebugStringHelper.h"

#include "pmlc/target/x86/pass_detail.h"
#include "pmlc/target/x86/passes.h"
#include "pmlc/util/logging.h"

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

  ValueRange values = op.getOperands();
  SmallVector<AffineMap, 3> idxMaps =
      llvm::to_vector<3>(op.indexing_maps().getAsValueRange<AffineMapAttr>());

  Block *block = op.getBody();
  Block::BlockArgListType args = block->getArguments();
  if (args.size() != 3)
    return None;

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

  if (matchPattern(
          yieldOp,
          m_Op<linalg::YieldOp>(m_Op<AddFOp>(
              m_Val(args[2]), m_Op<MulFOp>(m_Val(args[1]), m_Val(args[0]))))) ||
      matchPattern(yieldOp,
                   m_Op<linalg::YieldOp>(m_Op<AddFOp>(m_Op<MulFOp>(
                                             m_Val(args[1]), m_Val(args[0]))),
                                         m_Val(args[2]))))
    return ConvCapture{values, idxMaps, {1, 0, 2}};

  return None;
}

static AffineExpr getBlockedExpr(MLIRContext *context, int64_t hiDim,
                                 int64_t loDim, int64_t blockSize) {
  return getAffineDimExpr(hiDim, context) * blockSize +
         getAffineDimExpr(loDim, context);
}

struct ReorderLayoutsPass : public ReorderLayoutsBase<ReorderLayoutsPass> {
  void runOnFunction() final {
    getFunction().walk([&](linalg::GenericOp op) { reorderGenericOp(op); });
  }

  void reorderGenericOp(linalg::GenericOp op) {
    constexpr int64_t blockSize = 16;

    Optional<ConvCapture> conv = detectConv(op);
    if (!conv)
      return;

    // IVLOG(1, "Conv: " << debugString(op));

    // dims: (n, h, w, c, r, s, k)
    Optional<SmallVector<int64_t, 4>> ranges = op.getStaticLoopRanges();
    if (!ranges) {
      op.emitError("GenericOp does not have static ranges.");
      return;
    }

    if (ranges->size() != 7 ||           //
        (*ranges)[3] % blockSize != 0 || // C
        (*ranges)[6] % blockSize != 0)   // K
      return;

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
        createReorderLoop(builder, 5, blockedInputType, conv->input.value,
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
        createReorderLoop(builder, 6, blockedFilterType, conv->filter.value,
                          filterSourceMap, filterSinkMap);

    // Adjust convolution
    RankedTensorType blockedOutputType = conv->getBlockedOutputType(blockSize);
    auto init = builder.create<linalg::InitTensorOp>(
        blockedOutputType.getShape(), blockedOutputType.getElementType());

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

    auto newConv = builder.create<linalg::GenericOp>(
        TypeRange{blockedOutputType},
        ValueRange{reorderInput.getResult(0), reorderFilter.getResult(0)},
        ValueRange{init},
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
        createReorderLoop(builder, 5, conv->output.type, newConv.getResult(0),
                          inputSinkMap, inputSourceMap);

    op.getResult(0).replaceAllUsesWith(reorderOutput.getResult(0));
  }

  linalg::GenericOp createReorderLoop(ImplicitLocOpBuilder &builder,
                                      int64_t numDims,
                                      RankedTensorType resultType, Value source,
                                      AffineMap sourceMap, AffineMap sinkMap) {
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
