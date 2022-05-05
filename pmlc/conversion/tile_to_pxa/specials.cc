// Copyright 2021, Intel Corporation

#include "pmlc/conversion/tile_to_pxa/pass_detail.h"
#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/util/logging.h"

namespace pmlc::conversion::tile_to_pxa {

namespace layer = dialect::layer;
namespace pxa = dialect::pxa;
namespace tile = dialect::tile;

using namespace mlir; // NOLINT

using util::GatherMode;
using util::InterpolationMode;
using util::NearestMode;
using util::OutOfBoundsMode;
using util::ScatterMode;

namespace {

struct ArgSortOpConversion : public OpConversionPattern<tile::ArgSortOp> {
  using OpConversionPattern<tile::ArgSortOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::ArgSortOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    IVLOG(3, "ArgSortOpConversion::matchAndRewrite");
    Location loc = op.getLoc();
    IntegerType i32Type = rewriter.getI32Type();
    IndexType indexType = rewriter.getIndexType();

    // Axis represents one of the tensor dimensions.
    int64_t axisAttr = op.axis().getSExtValue();

    // Special case value -1 indicates the last axis.
    Value tensor = adaptor.tensor();
    ArrayRef<int64_t> shape = tensor.getType().cast<MemRefType>().getShape();
    size_t tensorDims = shape.size();
    if (axisAttr < 0) {
      axisAttr += static_cast<int64_t>(tensorDims);
    }
    size_t axis = static_cast<size_t>(axisAttr);
    if (axisAttr < 0 || axis >= tensorDims) {
      return failure();
    }
    Type elementType = tensor.getType().cast<MemRefType>().getElementType();

    // Allocate an output tensor to contain the sorted argument indices.
    MemRefType resultType = MemRefType::get(shape, i32Type);
    Value result =
        rewriter.create<memref::AllocOp>(loc, resultType).getResult();

    llvm::SmallVector<Type, 4> resultTypes;
    resultTypes.push_back(resultType);
    llvm::SmallVector<NamedAttribute, 4> attrs;
    llvm::SmallVector<Value, 2> layerInputs{result, tensor};
    auto layerOp =
        rewriter.create<layer::BoxOp>(loc, "argsort", layerInputs, resultTypes,
                                      rewriter.getDictionaryAttr(attrs));
    rewriter.setInsertionPointToStart(&layerOp.body().front());

    // Inside the box, the output tensor comes from the first argument,
    // and the input data tensor comes from the next. Outputs must come
    // first in the layer.box operands list.
    result = layerOp.body().getArguments()[0];
    tensor = layerOp.body().getArguments()[1];

    Value icon0 =
        rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(0)).getResult();
    Value icon1 =
        rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(1)).getResult();

    // Create a loop nest over all the dimensions, including the one we are
    // sorting on. We will use a single iteration for the sort axis, since the
    // body of the nest will contain the sorting loop, but we will keep the
    //  number of IVs equal to the tensor rank to simplify accounting.
    SmallVector<Value, 4> ops;
    for (size_t i = 0; i < tensorDims; ++i) {
      if (i == axis) {
        ops.push_back(icon0);
        continue;
      }
      Value limit =
          rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(shape[i]))
              .getResult();
      auto loop = rewriter.create<scf::ForOp>(loc, icon0, limit, icon1);
      ops.push_back(loop.getInductionVar());
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    Value iconN =
        rewriter.create<ConstantOp>(loc, rewriter.getIndexAttr(shape[axis]))
            .getResult();

    // For loads and stores from the minIdxVar and minValVar, we'll need a
    // single, zero-value index, because that's the only way to create a memref
    SmallVector<Value, 1> zeroIndex;
    zeroIndex.push_back(icon0);

    // Initialize result tensor using index values in ascending order
    {
      auto initLoop = rewriter.create<scf::ForOp>(loc, icon0, iconN, icon1);
      Value initIV = initLoop.getInductionVar();
      rewriter.setInsertionPointToStart(initLoop.getBody());
      // The induction var is the index value across the sort axis.
      // Write the value of the index to its position in the sorted output.
      // This will be our initial argument index into the source tensor.
      // Combine the index value with the loop dimension indexes to create the
      // destination affine map.
      Value indexVal =
          rewriter.create<arith::IndexCastOp>(loc, initIV, i32Type).getResult();
      ops[axis] = initIV;
      rewriter.create<memref::StoreOp>(loc, indexVal, result, ops);
      rewriter.setInsertionPointAfter(initLoop);
    }

    // Stable selection sort:
    // for (int i = 0; i < n-1; i++) {
    //     int min_idx = i;
    //     for (int j = i+1; j < n; j++) {
    //         if (arr[j] < arr[min_idx]) {
    //             min_idx = j;
    //         }
    //     }
    //     int min_val = arr[min_idx];
    //     while (min_idx > i) {
    //         arr[min_idx] = arr[min_idx - 1];
    //         min_idx--;
    //     }
    //     arr[i] = min_val;
    // }

    // Build inner sorting loop
    if (shape[axis] > 1) {
      auto sortUB = rewriter.create<ConstantOp>(
          loc, rewriter.getIndexAttr(shape[axis] - 1));
      auto sortLoop = rewriter.create<scf::ForOp>(loc, icon0, sortUB, icon1);
      Value sortIV = sortLoop.getInductionVar();
      rewriter.setInsertionPointToStart(sortLoop.getBody());

      // Get the value associated with the current minimum index position.
      // create a MemRefType which is a single-element index
      MemRefType minIdxVarType = MemRefType::get({1}, indexType);
      auto minIdxVar = rewriter.create<memref::AllocaOp>(loc, minIdxVarType);
      rewriter.create<memref::StoreOp>(loc, sortIV, minIdxVar, zeroIndex);

      ops[axis] = sortIV;
      Value minValIdxInt =
          rewriter.create<memref::LoadOp>(loc, result, ops).getResult();
      Value minValIdx =
          rewriter.create<arith::IndexCastOp>(loc, minValIdxInt, indexType)
              .getResult();
      ops[axis] = minValIdx;
      Value minVal =
          rewriter.create<memref::LoadOp>(loc, tensor, ops).getResult();
      MemRefType minValVarType = MemRefType::get({1}, elementType);
      auto minValVar = rewriter.create<memref::AllocaOp>(loc, minValVarType);
      rewriter.create<memref::StoreOp>(loc, minVal, minValVar, zeroIndex);

      // Iterate over the remaining elements, looking for a smaller value.
      auto minLB = rewriter.create<arith::AddIOp>(loc, sortIV, icon1);
      auto minLoop = rewriter.create<scf::ForOp>(loc, minLB, iconN, icon1);
      Value minIV = minLoop.getInductionVar();
      rewriter.setInsertionPointToStart(minLoop.getBody());

      // Get the comparison value for the search iterator position.
      ops[axis] = minIV;
      Value compValIdxInt =
          rewriter.create<memref::LoadOp>(loc, result, ops).getResult();
      Value compValIdx =
          rewriter.create<arith::IndexCastOp>(loc, compValIdxInt, indexType)
              .getResult();
      ops[axis] = compValIdx;
      Value compVal =
          rewriter.create<memref::LoadOp>(loc, tensor, ops).getResult();

      // What is the current minimum value, stored in the min val var?
      minVal = rewriter.create<memref::LoadOp>(loc, minValVar, zeroIndex)
                   .getResult();
      // Is compVal smaller than minVal? If so, update the allocs
      Value orderPred;
      orderPred = convertCmpOp(loc, rewriter, compVal, minVal, elementType,
                               op.direction());

      auto ifReorder = rewriter.create<scf::IfOp>(loc, orderPred, false);
      rewriter.setInsertionPointToStart(&ifReorder.getThenRegion().front());
      // store minIV -> minIdxVar
      rewriter.create<memref::StoreOp>(loc, minIV, minIdxVar, zeroIndex);
      // store compVal -> minValVar
      rewriter.create<memref::StoreOp>(loc, compVal, minValVar, zeroIndex);
      // End the conditional block. We would set the insertion point after
      // the ifReorder block, except that we are about to...
      // End the inner sort loop, which found the smallest value in the
      // unsorted region, by setting the insertion point after its block.
      rewriter.setInsertionPointAfter(minLoop);

      Value finalMinPos =
          rewriter.create<memref::LoadOp>(loc, minIdxVar, zeroIndex)
              .getResult();
      ops[axis] = finalMinPos;
      Value finalMinIdx =
          rewriter.create<memref::LoadOp>(loc, result, ops).getResult();

      // Move every element between [sortIV,finalMinPos) a step forward
      // and then move the minimum index to the head to keep it stable.
      auto moveLoop =
          rewriter.create<scf::ForOp>(loc, sortIV, finalMinPos, icon1);
      Value moveIV = moveLoop.getInductionVar();
      rewriter.setInsertionPointToStart(moveLoop.getBody());
      moveIV = rewriter.create<arith::SubIOp>(loc, finalMinPos, moveIV);
      moveIV = rewriter.create<arith::AddIOp>(loc, moveIV, sortIV);
      auto moveNext = rewriter.create<arith::SubIOp>(loc, moveIV, icon1);
      ops[axis] = moveNext;
      Value moveNextVal =
          rewriter.create<memref::LoadOp>(loc, result, ops).getResult();
      ops[axis] = moveIV;
      rewriter.create<memref::StoreOp>(loc, moveNextVal, result, ops);
      rewriter.setInsertionPointAfter(moveLoop);
      ops[axis] = sortIV;
      rewriter.create<memref::StoreOp>(loc, finalMinIdx, result, ops);

      // The minimum remaining value is now at the end of the sorted region.
      // Advance to the next minimum in the remaining unsorted region.
      rewriter.setInsertionPointAfter(sortLoop);
    }

    // for future reference, when N is a power of 2
    // bitonic sort:
    // for (int k = 2; k <= N; k = 2 * k) {
    //   for (int j = k >> 1; j > 0; j = j >> 1) {
    //     for (int i = 0; i < N; i++) {
    //       int ixj = i ^ j;
    //       if (ixj > i) {
    //         if ((i & k) == 0 && a[i] > a[ixj]) {
    //           swap(a[i], a[ixj]);
    //         }
    //         if ((i & k) != 0 && a[i] < a[ixj]) {
    //           swap(a[i], a[ixj]);
    //         }
    //       }
    //     }
    //   }
    // }

    rewriter.setInsertionPointToEnd(&layerOp.body().back());
    rewriter.create<layer::ReturnOp>(loc, ArrayRef<Value>{result});
    rewriter.replaceOp(op, layerOp.getResult(0));
    return success();
  }

  Value convertCmpOp(Location loc, ConversionPatternRewriter &rewriter,
                     Value compVal, Value minVal, Type type,
                     tile::SortDirection dir) const {
    if (type.isa<FloatType>()) {
      arith::CmpFPredicate pred;
      switch (dir) {
      case tile::SortDirection::asc:
        pred = arith::CmpFPredicate::OLT;
        break;
      case tile::SortDirection::desc:
        pred = arith::CmpFPredicate::OGT;
        break;
      }
      return rewriter.create<mlir::arith::CmpFOp>(loc, pred, compVal, minVal)
          .getResult();
    }
    arith::CmpIPredicate pred;
    if (type.isSignedInteger()) {
      switch (dir) {
      case tile::SortDirection::asc:
        pred = arith::CmpIPredicate::slt;
        break;
      case tile::SortDirection::desc:
        pred = arith::CmpIPredicate::sgt;
        break;
      }
    } else {
      switch (dir) {
      case tile::SortDirection::asc:
        pred = arith::CmpIPredicate::ult;
        break;
      case tile::SortDirection::desc:
        pred = arith::CmpIPredicate::ugt;
        break;
      }
    }
    return rewriter.create<mlir::arith::CmpIOp>(loc, pred, compVal, minVal)
        .getResult();
  }
};

struct GatherOpConversion : public OpConversionPattern<tile::GatherOp> {
  using OpConversionPattern<tile::GatherOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    // Input values
    Value tensor = adaptor.tensor();
    // Index values for the last dimension
    // this is a one-dimensional array of integers
    Value indices = adaptor.indices();

    TileToPXATypeConverter typeConverter;
    Type resultType = typeConverter.convertType(op.result().getType());
    auto memrefType = resultType.cast<MemRefType>();

    auto elementType = memrefType.getElementType();
    // Make an allocation for the output
    auto resultMemRef =
        rewriter.create<memref::AllocOp>(loc, memrefType).getResult();

    // We need an array of int64_t representing the results tensor's dims
    ArrayRef<int64_t> size = memrefType.getShape();

    auto loop = rewriter.create<AffineParallelOp>(
        loc, ArrayRef<Type>{memrefType},
        ArrayRef<arith::AtomicRMWKind>{arith::AtomicRMWKind::assign}, size);
    rewriter.setInsertionPointToStart(loop.getBody());

    // Create an affine map for loading the index, using the leading counters
    size_t axis = *(op.axis().getRawData());
    size_t batchDims = *(op.batchDims().getRawData());
    ArrayRef<int64_t> idxShape =
        indices.getType().cast<MemRefType>().getShape();
    size_t idxDims = idxShape.size();
    AffineMap idxLoadMap = AffineMap::getMultiDimIdentityMap(idxDims, ctx);

    // Create default source map
    size_t dstDims = size.size();
    std::vector<Value> srcOps;
    Value interpVal;
    switch (op.mode()) {
    case GatherMode::normal: {
      ArrayRef<BlockArgument> idxLoadOps = loop.getIVs().slice(axis, idxDims);
      Value idx =
          rewriter.create<pxa::PxaLoadOp>(loc, indices, idxLoadMap, idxLoadOps)
              .getResult();
      for (size_t i = 0; i < axis; ++i) {
        srcOps.push_back(loop.getIVs()[i]);
      }

      for (size_t i = axis + idxDims - 1; i < dstDims; ++i) {
        srcOps.push_back(loop.getIVs()[i]);
      }

      // Create std ops for 1D interpolation
      if (idx.getType().isa<FloatType>()) {
        if (elementType.isa<IntegerType>()) {
          idx = rewriter.create<mlir::arith::FPToSIOp>(loc, idx, rewriter.getI32Type())
                    .getResult();
          IndexType indexType = rewriter.getIndexType();
          // Cast from whatever integer type it has to index type
          idx = rewriter.create<mlir::arith::IndexCastOp>(loc, idx, indexType)
                    .getResult();
          srcOps.at(axis) = idx;
          interpVal = rewriter.create<memref::LoadOp>(loc, tensor, srcOps);
        } else {
          switch (op.interpolationMode()) {
          case InterpolationMode::nearest:
            interpVal = buildNearestInterpolationOps(
                loc, rewriter, tensor, idx, srcOps, axis, op.nearestMode(),
                op.OutOfBoundsMode());
            break;
          case InterpolationMode::linear:
            interpVal = buildLinearInterpolationOps(
                loc, rewriter, tensor, idx, srcOps, axis, op.OutOfBoundsMode());
            break;
          case InterpolationMode::cubic:
            interpVal = buildCubicInterpolationOps(
                loc, rewriter, tensor, idx, srcOps, axis,
                op.cubeCoeffAttr().getValueAsDouble(), op.OutOfBoundsMode());
            break;
          default:
            llvm_unreachable("Unsupported InterpolationMode");
          }
        }
      } else {
        if (!idx.getType().isa<IndexType>()) {
          IndexType indexType = rewriter.getIndexType();
          // Cast from whatever integer type it has to index type
          idx = rewriter.create<mlir::arith::IndexCastOp>(loc, idx, indexType)
                    .getResult();
        }
        srcOps.at(axis) = idx;
        interpVal = rewriter.create<memref::LoadOp>(loc, tensor, srcOps);
      }
    } break;
    case GatherMode::nd: {
      std::vector<Value> idxs, combIdx(idxDims);
      for (size_t i = 0; i < idxDims - 1; ++i) {
        combIdx[i] = loop.getIVs()[i];
      }
      for (int64_t i = 0; i < idxShape[idxDims - 1]; ++i) {
        combIdx[idxDims - 1] = rewriter.create<mlir::arith::ConstantIndexOp>(loc, i);
        Value idx =
            rewriter.create<pxa::PxaLoadOp>(loc, indices, idxLoadMap, combIdx)
                .getResult();
        if (!idx.getType().isa<IndexType>()) {
          IndexType indexType = rewriter.getIndexType();
          idx = rewriter.create<mlir::arith::IndexCastOp>(loc, idx, indexType)
                    .getResult();
        }
        idxs.push_back(idx);
      }
      for (size_t i = 0; i < batchDims; ++i) {
        srcOps.push_back(loop.getIVs()[i]);
      }
      srcOps.insert(srcOps.end(), idxs.begin(), idxs.end());
      for (size_t i = idxDims - 1; i < dstDims; ++i) {
        srcOps.push_back(loop.getIVs()[i]);
      }
      interpVal = rewriter.create<memref::LoadOp>(loc, tensor, srcOps);
    } break;
    default:
      llvm_unreachable("unrecognized gather mode");
    }

    // Create a destination map using all of the dimensions
    auto dstStoreMap = AffineMap::getMultiDimIdentityMap(dstDims, ctx);

    // Create a destination map from the whole loop
    auto stored = rewriter.create<pxa::PxaReduceOp>(loc, arith::AtomicRMWKind::assign,
                                                    interpVal, resultMemRef,
                                                    dstStoreMap, loop.getIVs());
    rewriter.create<AffineYieldOp>(loc, ArrayRef<Value>{stored.getResult()});
    rewriter.replaceOp(op, loop.getResult(0));
    return success();
  }

  Value buildNearestInterpolationOps(Location loc,
                                     ConversionPatternRewriter &rewriter,
                                     Value tensor, Value idx,
                                     std::vector<Value> &srcOps, size_t axis,
                                     NearestMode nearestMode,
                                     OutOfBoundsMode outOfBoundsMode) const {
    IndexType idxType = rewriter.getIndexType();
    IntegerType i32Type = rewriter.getI32Type();
    Type elementType = tensor.getType().cast<MemRefType>().getElementType();
    IndexBounds bounds = getIndexBounds(loc, rewriter, tensor, axis, i32Type);
    switch (nearestMode) {
    case NearestMode::round_prefer_floor: {
      Value cmp = isHalfWayFloat(loc, rewriter, idx);
      Value floor = floorFPToSI(loc, rewriter, idx, i32Type);
      Value round = roundFPToSI(loc, rewriter, idx, i32Type);
      idx = rewriter.create<arith::SelectOp>(loc, cmp, floor, round);
    } break;
    case NearestMode::round_prefer_ceil: {
      Value cmp = isHalfWayFloat(loc, rewriter, idx);
      Value ceil = ceilFPToSI(loc, rewriter, idx, i32Type);
      Value round = roundFPToSI(loc, rewriter, idx, i32Type);
      idx = rewriter.create<arith::SelectOp>(loc, cmp, ceil, round);
    } break;
    case NearestMode::floor:
      idx = floorFPToSI(loc, rewriter, idx, i32Type);
      break;
    case NearestMode::ceil:
      idx = ceilFPToSI(loc, rewriter, idx, i32Type);
      break;
    case NearestMode::simple:
      idx = rewriter.create<mlir::arith::FPToSIOp>(loc, idx, i32Type).getResult();
      break;
    default:
      llvm_unreachable("Unsupported NearestMode");
    }
    idx = checkIntOutOfBounds(loc, rewriter, idx, bounds);
    idx = rewriter.create<mlir::arith::IndexCastOp>(loc, idx, idxType).getResult();
    srcOps.at(axis) = idx;
    Value result = rewriter.create<memref::LoadOp>(loc, tensor, srcOps);
    result = processOutOfBoundsMode(loc, rewriter, result, idx, bounds,
                                    elementType, outOfBoundsMode);
    return result;
  }

  Value buildLinearInterpolationOps(Location loc,
                                    ConversionPatternRewriter &rewriter,
                                    Value tensor, Value idx,
                                    std::vector<Value> &srcOps, size_t axis,
                                    OutOfBoundsMode outOfBoundsMode) const {
    IndexType idxType = rewriter.getIndexType();
    IntegerType i32Type = rewriter.getI32Type();
    Type elementType = tensor.getType().cast<MemRefType>().getElementType();
    IndexBounds bounds = getIndexBounds(loc, rewriter, tensor, axis, i32Type);
    Value cst1F =
        rewriter
            .create<mlir::ConstantOp>(loc, elementType,
                                      rewriter.getFloatAttr(elementType, 1.0))
            .getResult();

    // Calculate interpolation nodes: floor and ceil
    Value floor = floorFPToSI(loc, rewriter, idx, i32Type);
    Value ceil = ceilFPToSI(loc, rewriter, idx, i32Type);
    floor = checkIntOutOfBounds(loc, rewriter, floor, bounds);
    ceil = checkIntOutOfBounds(loc, rewriter, ceil, bounds);
    floor = rewriter.create<mlir::arith::IndexCastOp>(loc, floor, idxType).getResult();
    ceil = rewriter.create<mlir::arith::IndexCastOp>(loc, ceil, idxType).getResult();

    // Load sample data g0 and g1 at interpolation nodes
    srcOps.at(axis) = ceil;
    Value g0 = rewriter.create<memref::LoadOp>(loc, tensor, srcOps).getResult();
    srcOps.at(axis) = floor;
    Value g1 = rewriter.create<memref::LoadOp>(loc, tensor, srcOps).getResult();

    // Calculate coefficients of g0 and g1
    Value floorF =
        rewriter.create<mlir::math::FloorOp>(loc, elementType, idx).getResult();
    Value c0 = rewriter.create<mlir::arith::SubFOp>(loc, idx, floorF).getResult();
    Value c1 = rewriter.create<mlir::arith::SubFOp>(loc, cst1F, c0).getResult();

    // Return interpolation result (result = c0*g0 + c1*g1)
    Value p0 = rewriter.create<mlir::arith::MulFOp>(loc, c0, g0).getResult();
    Value p1 = rewriter.create<mlir::arith::MulFOp>(loc, c1, g1).getResult();
    Value result = rewriter.create<mlir::arith::AddFOp>(loc, p0, p1).getResult();
    result = processOutOfBoundsMode(loc, rewriter, result, idx, bounds,
                                    elementType, outOfBoundsMode);
    return result;
  }

  Value buildCubicInterpolationOps(Location loc,
                                   ConversionPatternRewriter &rewriter,
                                   Value tensor, Value idx,
                                   std::vector<Value> &srcOps, size_t axis,
                                   double cubicCoeff,
                                   OutOfBoundsMode outOfBoundsMode) const {
    // Follow the algorithm used in ngraph cubic interpolation (also see, e.g.
    // [article](https://ieeexplore.ieee.org/document/1163711/).

    IndexType idxType = rewriter.getIndexType();
    IntegerType i32Type = rewriter.getI32Type();
    Type elementType = tensor.getType().cast<MemRefType>().getElementType();
    IndexBounds bounds = getIndexBounds(loc, rewriter, tensor, axis, i32Type);

    // Create constant a (cubeCoeff)
    Value a = rewriter
                  .create<mlir::ConstantOp>(
                      loc, rewriter.getF64Type(),
                      FloatAttr::get(rewriter.getF64Type(), cubicCoeff))
                  .getResult();
    if (!elementType.isa<mlir::Float64Type>()) {
      a = rewriter.create<mlir::arith::TruncFOp>(loc, elementType, a);
    }

    // Create integer constants
    SmallVector<Value, 4> cstI;
    for (size_t i = 0; i <= 2; i++) {
      auto cstOp = rewriter.create<mlir::ConstantOp>(
          loc, i32Type, rewriter.getIntegerAttr(i32Type, i));
      cstI.push_back(cstOp.getResult());
    }

    // Create float constants
    SmallVector<Value, 4> cstF;
    for (size_t i = 0; i <= 3; i++) {
      auto cstOp = rewriter.create<mlir::ConstantOp>(
          loc, elementType, rewriter.getFloatAttr(elementType, i));
      cstF.push_back(cstOp.getResult());
    }

    // Calculate interpolation nodes x
    Value floorI = floorFPToSI(loc, rewriter, idx, i32Type);
    Value ceilI = ceilFPToSI(loc, rewriter, idx, i32Type);
    SmallVector<Value, 4> x;
    x.push_back(
        rewriter.create<mlir::arith::SubIOp>(loc, floorI, cstI[1]).getResult());
    x.push_back(floorI);
    x.push_back(ceilI);
    x.push_back(rewriter.create<mlir::arith::AddIOp>(loc, ceilI, cstI[1]).getResult());

    // Load sample data g at interpolation nodes
    SmallVector<Value, 4> g;
    for (size_t i = 0; i < x.size(); i++) {
      x[i] = checkIntOutOfBounds(loc, rewriter, x[i], bounds);
      x[i] = rewriter.create<mlir::arith::IndexCastOp>(loc, x[i], idxType).getResult();
      srcOps.at(axis) = x[i];
      auto loadOp = rewriter.create<memref::LoadOp>(loc, tensor, srcOps);
      g.push_back(loadOp.getResult());
    }

    // Calculate intermediate terms
    SmallVector<Value, 4> p;
    Value floorF =
        rewriter.create<mlir::math::FloorOp>(loc, idx.getType(), idx).getResult();
    Value s = rewriter.create<mlir::arith::SubFOp>(loc, idx, floorF).getResult();
    Value s2 = rewriter.create<mlir::arith::MulFOp>(loc, s, s).getResult();
    Value s3 = rewriter.create<mlir::arith::MulFOp>(loc, s2, s).getResult();
    Value s_a = rewriter.create<mlir::arith::MulFOp>(loc, a, s).getResult();
    Value s2_a = rewriter.create<mlir::arith::MulFOp>(loc, a, s2).getResult();
    Value s3_a = rewriter.create<mlir::arith::MulFOp>(loc, a, s3).getResult();
    Value s3_a2 = rewriter.create<mlir::arith::AddFOp>(loc, a, cstF[2]).getResult();
    s3_a2 = rewriter.create<mlir::arith::MulFOp>(loc, s3_a2, s3).getResult();
    Value s2_a3 = rewriter.create<mlir::arith::AddFOp>(loc, a, cstF[3]).getResult();
    s2_a3 = rewriter.create<mlir::arith::MulFOp>(loc, s2_a3, s2).getResult();
    Value s2_2a3 = rewriter.create<mlir::arith::AddFOp>(loc, a, a).getResult();
    s2_2a3 = rewriter.create<mlir::arith::AddFOp>(loc, s2_2a3, cstF[3]).getResult();
    s2_2a3 = rewriter.create<mlir::arith::MulFOp>(loc, s2_2a3, s2).getResult();

    // Calculate 4 terms at interpolation nodes
    p.push_back(rewriter.create<mlir::arith::MulFOp>(loc, s2_a, cstF[2]).getResult());
    p[0] = rewriter.create<mlir::arith::SubFOp>(loc, s3_a, p[0]).getResult();
    p[0] = rewriter.create<mlir::arith::AddFOp>(loc, p[0], s_a).getResult();

    p.push_back(rewriter.create<mlir::arith::SubFOp>(loc, s3_a2, s2_a3).getResult());
    p[1] = rewriter.create<mlir::arith::AddFOp>(loc, p[1], cstF[1]).getResult();

    p.push_back(rewriter.create<mlir::arith::SubFOp>(loc, s2_2a3, s3_a2).getResult());
    p[2] = rewriter.create<mlir::arith::SubFOp>(loc, p[2], s_a).getResult();

    p.push_back(rewriter.create<mlir::arith::SubFOp>(loc, s2_a, s3_a).getResult());

    for (size_t i = 0; i < p.size(); i++) {
      p[i] = rewriter.create<mlir::arith::MulFOp>(loc, p[i], g[i]).getResult();
    }

    // Return interpolation result (result = p0 + p1 + p2 + p3)
    Value result = rewriter.create<mlir::arith::AddFOp>(loc, p[0], p[1]).getResult();
    result = rewriter.create<mlir::arith::AddFOp>(loc, result, p[2]).getResult();
    result = rewriter.create<mlir::arith::AddFOp>(loc, result, p[3]).getResult();

    result = processOutOfBoundsMode(loc, rewriter, result, idx, bounds,
                                    elementType, outOfBoundsMode);
    return result;
  }

  struct IndexBounds {
    Value lower;
    Value upper;
  };

  IndexBounds getIndexBounds(Location loc, ConversionPatternRewriter &rewriter,
                             Value tensor, size_t axis,
                             IntegerType integerType) const {
    // Return lower and upper bounds of a tensor at an axis
    SmallVector<Value, 2> bounds;
    int64_t axisLen = tensor.getType().cast<MemRefType>().getShape()[axis];
    auto lower = rewriter.create<mlir::ConstantOp>(
        loc, integerType, rewriter.getIntegerAttr(integerType, 0));
    auto upper = rewriter.create<mlir::ConstantOp>(
        loc, integerType, rewriter.getIntegerAttr(integerType, axisLen - 1));
    return IndexBounds{lower, upper};
  }

  Value checkIntOutOfBounds(Location loc, ConversionPatternRewriter &rewriter,
                            Value value, const IndexBounds &bounds) const {
    // Check if a mlir::IntegerType value is out of bounds. If it is, set it to
    // lower/upper bound.
    auto cmpLower = rewriter.create<mlir::arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                  value, bounds.lower);
    auto cmpUpper = rewriter.create<mlir::arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                  value, bounds.upper);
    value = rewriter.create<arith::SelectOp>(loc, cmpLower, bounds.lower, value);
    value = rewriter.create<arith::SelectOp>(loc, cmpUpper, value, bounds.upper);
    return value;
  }

  Value isHalfWayFloat(Location loc, ConversionPatternRewriter &rewriter,
                       Value value) const {
    // Check if the fractional part of a float value is 0.5
    Type floatType = value.getType();
    auto half = rewriter.create<mlir::ConstantOp>(
        loc, floatType, rewriter.getFloatAttr(floatType, 0.5));
    Value floor =
        rewriter.create<mlir::math::FloorOp>(loc, floatType, value).getResult();
    auto floorPlusHalf = rewriter.create<mlir::arith::AddFOp>(loc, floor, half);
    return rewriter
        .create<mlir::arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, value, floorPlusHalf)
        .getResult();
  }

  Value ceilFPToSI(Location loc, ConversionPatternRewriter &rewriter,
                   Value value, IntegerType integerType) const {
    Value ceilFloat =
        rewriter.create<mlir::math::CeilOp>(loc, value.getType(), value).getResult();
    return rewriter.create<mlir::arith::FPToSIOp>(loc, integerType, ceilFloat)
        .getResult();
  }

  Value floorFPToSI(Location loc, ConversionPatternRewriter &rewriter,
                    Value value, IntegerType integerType) const {
    Value floorFloat =
        rewriter.create<mlir::math::FloorOp>(loc, value.getType(), value)
            .getResult();
    return rewriter.create<mlir::arith::FPToSIOp>(loc, integerType, floorFloat)
        .getResult();
  }

  Value roundFPToSI(Location loc, ConversionPatternRewriter &rewriter,
                    Value value, IntegerType integerType) const {
    Type floatType = value.getType();
    auto half = rewriter.create<mlir::ConstantOp>(
        loc, floatType, rewriter.getFloatAttr(floatType, 0.5));
    auto valuePlusHalf = rewriter.create<mlir::arith::AddFOp>(loc, value, half);
    return floorFPToSI(loc, rewriter, valuePlusHalf, integerType);
  }

  Value processOutOfBoundsMode(Location loc,
                               ConversionPatternRewriter &rewriter,
                               Value result, Value idx, IndexBounds bounds,
                               Type elementType,
                               OutOfBoundsMode outOfBoundsMode) const {
    switch (outOfBoundsMode) {
    case OutOfBoundsMode::gather_edge_padded_input:
      break;
    case OutOfBoundsMode::return_zero: {
      auto zero =
          rewriter
              .create<mlir::ConstantOp>(loc, elementType,
                                        rewriter.getFloatAttr(elementType, 0.0))
              .getResult();
      auto bound_upper =
          rewriter.create<mlir::arith::SIToFPOp>(loc, bounds.upper, elementType)
              .getResult();
      auto bound_lower =
          rewriter.create<mlir::arith::SIToFPOp>(loc, bounds.lower, elementType)
              .getResult();
      auto cmp_upper = rewriter.create<mlir::arith::CmpFOp>(loc, arith::CmpFPredicate::OGT,
                                                     idx, bound_upper);
      auto cmp_lower = rewriter.create<mlir::arith::CmpFOp>(loc, arith::CmpFPredicate::OLT,
                                                     idx, bound_lower);
      result = rewriter.create<arith::SelectOp>(loc, cmp_upper, zero, result);
      result = rewriter.create<arith::SelectOp>(loc, cmp_lower, zero, result);
    } break;
    default:
      llvm_unreachable("Unsupported OutOfBoundsMode");
    }
    return result;
  }
};

struct ScatterOpConversion : public OpConversionPattern<tile::ScatterOp> {
  using OpConversionPattern<tile::ScatterOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::ScatterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // Helpful explanation of scatter from tensorflow docs:
    // https://www.tensorflow.org/api_docs/python/tf/scatter_nd

    Location loc = op.getLoc();
    MLIRContext *ctx = rewriter.getContext();
    TileToPXATypeConverter typeConverter;
 
    Value data = adaptor.data();
    Value indices = adaptor.indices();
    Value updates = adaptor.updates();

    // Make an allocation for the output
    Type resultType = typeConverter.convertType(op.result().getType());
    auto resultMemRefType = resultType.cast<MemRefType>();
    Value resultMemRef =
        rewriter.create<memref::AllocOp>(loc, resultMemRefType).getResult();

    ArrayRef<int64_t> dataShape = data.getType().cast<MemRefType>().getShape();
    auto copyLoop = rewriter.create<AffineParallelOp>(
        loc, ArrayRef<Type>{data.getType()},
        ArrayRef<arith::AtomicRMWKind>{arith::AtomicRMWKind::assign}, dataShape);
    rewriter.setInsertionPointToStart(copyLoop.getBody());
    size_t dataDims = dataShape.size();
    AffineMap dataLoadMap = AffineMap::getMultiDimIdentityMap(dataDims, ctx);
    auto loadData = rewriter.create<pxa::PxaLoadOp>(loc, data, dataLoadMap,
                                                    copyLoop.getIVs());
    auto stored = buildSimpleStore(rewriter, loc, loadData, resultMemRef,
                                   tile::getPaddingInfo(op));

    rewriter.create<AffineYieldOp>(loc, ArrayRef<Value>{stored});
    rewriter.setInsertionPointAfter(copyLoop);

    // Get the shape of the update tensor and create a parallel loop over its
    // indexes; we will load each value from the updates, load its destination
    // from the indexes, and store the value to the result.
    Type updatesType = typeConverter.convertType(updates.getType());
    auto updatesMemRefType = updatesType.cast<MemRefType>();
    ArrayRef<int64_t> updatesShape = updatesMemRefType.getShape();

    auto loop = rewriter.create<AffineParallelOp>(
        loc, ArrayRef<Type>{resultMemRefType},
        ArrayRef<arith::AtomicRMWKind>{arith::AtomicRMWKind::assign}, updatesShape);
    rewriter.setInsertionPointToStart(loop.getBody());

    // Load the source value from the updates tensor.
    // The affine map for locating the update value uses all loop dimensions.
    size_t srcDims = updatesShape.size();
    AffineMap srcLoadMap = AffineMap::getMultiDimIdentityMap(srcDims, ctx);
    MutableArrayRef<BlockArgument> srcLoadOps = loop.getIVs();
    Value srcVal =
        rewriter.create<pxa::PxaLoadOp>(loc, updates, srcLoadMap, srcLoadOps)
            .getResult();

    // Load the location value from the indices tensor.
    // Create an affine map for loading the index, using leading counters.
    size_t axis = *(op.axis().getRawData());
    ArrayRef<int64_t> idxShape =
        indices.getType().cast<MemRefType>().getShape();
    size_t idxDims = idxShape.size();
    AffineMap idxLoadMap = AffineMap::getMultiDimIdentityMap(idxDims, ctx);
    SmallVector<Value, 4> dstOps;

    switch (op.mode()) {
    case ScatterMode::update_nd: {
      std::vector<Value> idxs, combIdx(idxDims);
      for (size_t i = 0; i < idxDims - 1; ++i) {
        combIdx[i] = loop.getIVs()[i];
      }
      for (int64_t i = 0; i < idxShape[idxDims - 1]; ++i) {
        combIdx[idxDims - 1] = rewriter.create<mlir::arith::ConstantIndexOp>(loc, i);
        Value indexVal =
            getIndexValue(loc, rewriter, indices, idxLoadMap, combIdx);
        idxs.push_back(indexVal);
      }
      dstOps.insert(dstOps.begin(), idxs.begin(), idxs.end());
      for (size_t i = idxDims - 1; i < srcDims; ++i) {
        dstOps.push_back(loop.getIVs()[i]);
      }
    } break;
    case ScatterMode::update_slice: {
      ArrayRef<BlockArgument> idxLoadOps = loop.getIVs().slice(axis, idxDims);
      size_t idxStart = axis + idxDims - 1;
      Value indexVal =
          getIndexValue(loc, rewriter, indices, idxLoadMap, idxLoadOps);
      getOutputIndices(indexVal, axis, idxStart, srcDims, dstOps,
                       loop.getIVs());
    } break;
    case ScatterMode::normal:
    case ScatterMode::update_elt: {
      ArrayRef<BlockArgument> idxLoadOps = loop.getIVs().take_front(idxDims);
      Value indexVal =
          getIndexValue(loc, rewriter, indices, idxLoadMap, idxLoadOps);
      getOutputIndices(indexVal, axis, /*idxStart=*/axis, srcDims, dstOps,
                       loop.getIVs());
    } break;
    default:
      llvm_unreachable("unrecognized scatter mode");
    }

    Value storeResult;
    if (op.mode() == ScatterMode::normal) {
      if (srcVal.getType().isa<FloatType>()) {
        storeResult = rewriter.create<pxa::PxaStoreOp>(
            loc, arith::AtomicRMWKind::addf, srcVal, copyLoop.getResult(0), dstOps);
      } else if (srcVal.getType().isa<IntegerType>()) {
        storeResult = rewriter.create<pxa::PxaStoreOp>(
            loc, arith::AtomicRMWKind::addi, srcVal, copyLoop.getResult(0), dstOps);
      } else {
        llvm_unreachable("Unsupported datatype in scatter.");
      }
    } else {
      storeResult = rewriter.create<pxa::PxaStoreOp>(
          loc, arith::AtomicRMWKind::assign, srcVal, copyLoop.getResult(0), dstOps);
    }

    rewriter.create<AffineYieldOp>(loc, ArrayRef<Value>{storeResult});
    rewriter.replaceOp(op, loop.getResult(0));
    return success();
  }

  Value getIndexValue(Location loc, ConversionPatternRewriter &rewriter,
                      Value indices, AffineMap idxLoadMap,
                      mlir::ValueRange idxLoadOps) const {
    Value indexVal =
        rewriter.create<pxa::PxaLoadOp>(loc, indices, idxLoadMap, idxLoadOps)
            .getResult();

    // Cast the index value from its integer type to the index type
    if (!indexVal.getType().isa<IndexType>()) {
      // cast from whatever integer type it has to index type
      IndexType indexType = rewriter.getIndexType();
      indexVal = rewriter.create<mlir::arith::IndexCastOp>(loc, indexVal, indexType)
                     .getResult();
    }
    return indexVal;
  }

  void getOutputIndices(Value indexVal, size_t axis, size_t idxStart,
                        size_t end, SmallVector<Value, 4> &dstOps,
                        ArrayRef<BlockArgument> loopArgs) const {
    for (size_t i = 0; i < axis; ++i) {
      dstOps.push_back(loopArgs[i]);
    }

    for (size_t i = idxStart; i < end; ++i) {
      dstOps.push_back(loopArgs[i]);
    }

    dstOps[axis] = indexVal;
  }
};

struct PrngOpConversion : public OpConversionPattern<tile::PrngOp> {
  using OpConversionPattern<tile::PrngOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tile::PrngOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    tile::PrngOpAdaptor transformed(adaptor.getOperands());
    BufferAllocator allocResult(rewriter, op.getOperation(),
                                op.result().getType());
    BufferAllocator stateResult(rewriter, op.getOperation(),
                                op.state().getType());
    rewriter.replaceOpWithNewOp<pxa::PrngOp>(
        op, allocResult.memRefType, stateResult.memRefType, transformed.state(),
        allocResult.resultMemRef, stateResult.resultMemRef);
    return success();
  }
};

} // namespace

void populateTileToPXASpecialPatterns(mlir::RewritePatternSet &patterns) {
  patterns.insert<ArgSortOpConversion, //
                  GatherOpConversion,  //
                  PrngOpConversion,    //
                  ScatterOpConversion>(patterns.getContext());
}

} // namespace pmlc::conversion::tile_to_pxa
