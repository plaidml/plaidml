// Copyright 2020 Intel Corporation

#include "pmlc/dialect/pxa/transforms/cache.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "pmlc/dialect/pxa/analysis/strides.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/util/ident.h"
#include "pmlc/util/logging.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

static Value createInitLoop(OpBuilder &builder, Location loc, Value memref,
                            Value initVal) {
  auto memrefType = memref.getType().cast<MemRefType>();
  ArrayRef<int64_t> size = memrefType.getShape();
  assert(memrefType.getElementType() == initVal.getType());
  auto loop = builder.create<AffineParallelOp>(
      loc, ArrayRef<Type>{memrefType},
      ArrayRef<AtomicRMWKind>{AtomicRMWKind::assign}, size);
  auto initBuilder = loop.getBodyBuilder();
  auto idMap =
      AffineMap::getMultiDimIdentityMap(size.size(), builder.getContext());
  auto stored = initBuilder.create<PxaReduceOp>(
      loc, AtomicRMWKind::assign, initVal, memref, idMap, loop.getIVs());
  initBuilder.create<AffineYieldOp>(loc, ArrayRef<Value>{stored});
  return loop.getResult(0);
}

static Value createCopyLoop(OpBuilder &builder,               //
                            Location loc,                     //
                            ArrayRef<int64_t> size,           //
                            Value srcMemRef, Value dstMemRef, //
                            ArrayRef<StrideInfo> srcOffset,   //
                            ArrayRef<StrideInfo> dstOffset, AtomicRMWKind agg) {
  assert(size.size() == srcOffset.size());
  assert(size.size() == dstOffset.size());
  size_t dims = size.size();
  auto ctx = builder.getContext();
  auto loop = builder.create<AffineParallelOp>(
      loc, ArrayRef<Type>{dstMemRef.getType()},
      ArrayRef<AtomicRMWKind>{AtomicRMWKind::assign}, size);
  SmallVector<StrideInfo, 4> srcAccess;
  SmallVector<StrideInfo, 4> dstAccess;
  for (size_t i = 0; i < dims; i++) {
    auto offset = StrideInfo(loop.getIVs()[i]);
    srcAccess.push_back(srcOffset[i] + offset);
    dstAccess.push_back(dstOffset[i] + offset);
  }
  auto loadMap = convertToValueMap(ctx, srcAccess);
  auto reduceMap = convertToValueMap(ctx, dstAccess);
  auto txBuilder = loop.getBodyBuilder();
  auto loaded = txBuilder.create<PxaLoadOp>(
      loc, srcMemRef, loadMap.getAffineMap(), loadMap.getOperands());
  auto stored = txBuilder.create<PxaReduceOp>(loc, agg, loaded, dstMemRef,
                                              reduceMap.getAffineMap(),
                                              reduceMap.getOperands());
  txBuilder.create<AffineYieldOp>(loc, ArrayRef<Value>{stored});
  return loop.getResult(0);
}

LogicalResult cacheLoad(AffineParallelOp par, PxaLoadOp load) {
  // Get the striding information for the load op, fail if unsuccessful
  auto maybeRap = computeRelativeAccess(par.getBody(), load);
  if (!maybeRap) {
    return failure();
  }
  const auto &rap = *maybeRap;
  // Fail for non-1 strides, TODO: Handle compression case
  for (int64_t stride : rap.innerStride) {
    if (stride != 1) {
      return failure();
    }
  }
  // Prep for generation
  auto loc = load.getLoc();
  auto builder = OpBuilder::atBlockBegin(par.getBody());
  // Allocate a temporary buffer
  auto type =
      MemRefType::get(rap.innerCount, load.getMemRefType().getElementType());
  auto localBuf = builder.create<AllocOp>(loc, type);
  // Implement the copy loop
  SmallVector<StrideInfo, 4> zeroOffset(rap.innerCount.size());
  auto copy =
      createCopyLoop(builder, loc, rap.innerCount, load.getMemRef(), localBuf,
                     rap.outer, zeroOffset, AtomicRMWKind::assign);

  // Make a new load and remove the old one
  auto innerMap = convertToValueMap(par.getContext(), rap.inner);
  OpBuilder newLoadBuilder(load);
  auto newLoad = newLoadBuilder.create<PxaLoadOp>(
      loc, copy, innerMap.getAffineMap(), innerMap.getOperands());
  load.replaceAllUsesWith(newLoad.result());
  load.erase();

  return success();
}

LogicalResult cacheReduce(AffineParallelOp par, PxaReduceOp reduce) {
  // Get the striding information for the load op, fail if unsuccessful
  auto maybeRap = computeRelativeAccess(par.getBody(), reduce);
  if (!maybeRap) {
    return failure();
  }
  const auto &rap = *maybeRap;
  // Fail for non-1 strides, TODO: Handle compression case
  for (int64_t stride : rap.innerStride) {
    if (stride != 1) {
      return failure();
    }
  }
  // Compute the first use of the buffer written to by the reduce and verify
  // that there are only yields between
  Value out = reduce;
  while (out.getParentBlock() != par.getBody()) {
    if (!out.hasOneUse()) {
      return failure();
    }
    auto &use = *out.use_begin();
    auto yieldOp = dyn_cast<AffineYieldOp>(use.getOwner());
    if (!yieldOp) {
      return failure();
    }
    out = yieldOp.getParentOp()->getResult(use.getOperandNumber());
  }
  // Verify final output has a single use, TODO: THis probably isn't a hard
  // requirement, but it's easier
  if (!out.hasOneUse()) {
    return failure();
  }
  auto &finalUse = *out.use_begin();
  // Prep for generation
  auto loc = reduce.getLoc();
  auto builder = OpBuilder::atBlockBegin(par.getBody());
  // Allocate a temporary buffer
  auto eltType = reduce.getMemRefType().getElementType();
  auto type = MemRefType::get(rap.innerCount, eltType);
  auto localBuf = builder.create<AllocOp>(loc, type);
  // Clear it to the reduction identity
  auto ident = createIdentity(builder, loc, reduce.agg(), eltType);
  auto initBuf = createInitLoop(builder, loc, localBuf, ident);

  // Make a new load and remove the old one
  auto innerMap = convertToValueMap(par.getContext(), rap.inner);
  OpBuilder newReduceBuilder(reduce);
  auto newReduce = newReduceBuilder.create<PxaReduceOp>(
      loc, reduce.agg(), reduce.val(), initBuf, innerMap.getAffineMap(),
      innerMap.getOperands());
  reduce.replaceAllUsesWith(newReduce.result());
  // Walk upwards a second time and fix typing
  out = newReduce;
  Type newType = out.getType();
  while (out.getParentBlock() != par.getBody()) {
    auto &use = *out.use_begin();
    auto yieldOp = cast<AffineYieldOp>(use.getOwner());
    out = yieldOp.getParentOp()->getResult(use.getOperandNumber());
    out.setType(newType);
  }
  // Implement the copy loop
  builder.setInsertionPoint(par.getBody(), std::prev(par.getBody()->end()));
  SmallVector<StrideInfo, 4> zeroOffset(rap.innerCount.size());
  auto copyLoop =
      createCopyLoop(builder, loc, rap.innerCount, out, reduce.getMemRef(),
                     zeroOffset, rap.outer, reduce.agg());
  finalUse.set(copyLoop);
  reduce.erase();
  return success();
}

} // namespace pmlc::dialect::pxa
