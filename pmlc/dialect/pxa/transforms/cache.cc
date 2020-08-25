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
  auto loop = builder.create<AffineParallelOp>(loc, memrefType,
                                               AtomicRMWKind::assign, size);
  auto initBuilder = loop.getBodyBuilder();
  auto idMap =
      AffineMap::getMultiDimIdentityMap(size.size(), builder.getContext());
  auto stored = initBuilder.create<PxaReduceOp>(
      loc, AtomicRMWKind::assign, initVal, memref, idMap, loop.getIVs());
  initBuilder.create<AffineYieldOp>(loc, ArrayRef<Value>{stored});
  return loop.getResult(0);
}

static AffineParallelOp createCopyLoop(OpBuilder &builder,               //
                                       Location loc,                     //
                                       ArrayRef<int64_t> size,           //
                                       Value srcMemRef, Value dstMemRef, //
                                       ArrayRef<StrideInfo> srcOffset,   //
                                       ArrayRef<StrideInfo> dstOffset,
                                       AtomicRMWKind agg) {
  assert(size.size() == srcOffset.size());
  assert(size.size() == dstOffset.size());
  size_t dims = size.size();
  auto ctx = builder.getContext();
  auto loop = builder.create<AffineParallelOp>(loc, dstMemRef.getType(),
                                               AtomicRMWKind::assign, size);
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
  return loop;
}

Optional<CacheInfo> cacheLoad(AffineParallelOp par, PxaLoadOp load) {
  // Get the striding information for the load op, fail if unsuccessful
  auto maybeRap = computeRelativeAccess(load, par.getBody());
  if (!maybeRap) {
    IVLOG(3, "cacheLoad: Failed due to a non-strided access");
    return None;
  }
  const auto &rap = *maybeRap;
  // Fail for non-1 strides, TODO: Handle compression case
  for (int64_t stride : rap.innerStride) {
    if (stride != 1) {
      IVLOG(3, "cacheLoad failed: not all strides are one");
      return None;
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
  auto copyLoop =
      createCopyLoop(builder, loc, rap.innerCount, load.getMemRef(), localBuf,
                     rap.outer, zeroOffset, AtomicRMWKind::assign);

  // Make a new load and remove the old one
  auto innerMap = convertToValueMap(par.getContext(), rap.inner);
  OpBuilder newLoadBuilder(load);
  auto newLoad = newLoadBuilder.create<PxaLoadOp>(loc, copyLoop.getResult(0),
                                                  innerMap.getAffineMap(),
                                                  innerMap.getOperands());
  load.replaceAllUsesWith(newLoad.result());
  load.erase();

  return CacheInfo{
      /*copyLoopOp=*/copyLoop,
      /*relativeAccess=*/*maybeRap,
  };
}

LogicalResult cacheLoadAsVector(AffineParallelOp par, PxaLoadOp load,
                                int64_t reqVecSize) {
  // Get the striding information for the load op, fail if unsuccessful
  auto maybeRap = computeRelativeAccess(load, par.getBody());
  if (!maybeRap) {
    IVLOG(3, "cacheLoadAsVector: Failed due to a non-strided access");
    return failure();
  }
  const auto &rap = *maybeRap;
  // Require all sizes to be 1 except the final one, which is the vector size
  // and must be stride 1.
  if (rap.innerCount.size() < 1) {
    IVLOG(3, "cacheLoadAsVector: Failed due to 0-dim memref");
    return failure();
  }
  for (unsigned i = 0; i < rap.inner.size() - 1; i++) {
    if (rap.innerCount[i] != 1) {
      IVLOG(3, "cacheLoadAsVector: Failed due to invalid non-final count: "
                   << rap.innerCount[i]);
      return failure();
    }
  }
  unsigned last = rap.inner.size() - 1;
  if (rap.innerStride[last] != 1) {
    IVLOG(3, "cacheLoadAsVector: Failed due to invalid stride: "
                 << rap.innerStride[last]);
    return failure();
  }
  int64_t vectorSize = rap.innerCount[last];
  if (vectorSize == 1) {
    IVLOG(3, "cacheLoadAsVector: Failed due to size 1 vector width");
    return failure();
  }
  if (reqVecSize && vectorSize != reqVecSize) {
    IVLOG(3, "cacheLoadAsVector: Failed due to mismatch of required size");
    return failure();
  }
  auto eltType = load.getMemRefType().getElementType();
  auto vecType = VectorType::get({vectorSize}, eltType);
  // Prep for generation
  auto loc = load.getLoc();
  auto builder = OpBuilder::atBlockBegin(par.getBody());
  // Load as a vector
  auto loadMap = convertToValueMap(load.getContext(), rap.outer);
  IVLOG(2, "Making vector load");
  auto loadVec = builder.create<PxaVectorLoadOp>(loc, vecType, load.getMemRef(),
                                                 loadMap.getAffineMap(),
                                                 loadMap.getOperands());
  // Make a new load and remove the old one
  OpBuilder newLoadBuilder(load);
  // Do an affine apply to get the index
  auto innerMap = convertToValueMap(par.getContext(), rap.inner);
  // Extract the right element of the vector
  Value idx = newLoadBuilder.create<AffineApplyOp>(
      loc, innerMap.getAffineMap().getSubMap({last}), innerMap.getOperands());
  auto newLoad = newLoadBuilder.create<ExtractElementOp>(
      loc, eltType, loadVec.getResult(), idx);
  load.replaceAllUsesWith(newLoad.result());
  load.erase();
  return success();
}

Optional<CacheInfo> cacheReduce(AffineParallelOp par, PxaReduceOp reduce) {
  // Get the striding information for the load op, fail if unsuccessful
  auto maybeRap = computeRelativeAccess(reduce, par.getBody());
  if (!maybeRap) {
    IVLOG(3, "cacheReduce failed: due to a non-strided access");
    return None;
  }
  const auto &rap = *maybeRap;

  // Fail for non-1 strides, TODO: Handle compression case
  for (int64_t stride : rap.innerStride) {
    if (stride != 1) {
      IVLOG(3, "cacheReduce failed: not all strides are one.");
      return None;
    }
  }

  // Compute the first use of the buffer written to by the reduce and verify
  // that there are only yields between
  Value out = reduce;
  while (out.getParentBlock() != par.getBody()) {
    if (!out.hasOneUse()) {
      IVLOG(3, "cacheReduce failed: multiple uses");
      return None;
    }
    auto &use = *out.use_begin();
    auto yieldOp = dyn_cast<AffineYieldOp>(use.getOwner());
    if (!yieldOp) {
      IVLOG(3, "cacheReduce failed: missing yield op");
      return None;
    }
    out = yieldOp.getParentOp()->getResult(use.getOperandNumber());
  }

  // Verify final output has a single use, TODO: THis probably isn't a hard
  // requirement, but it's easier
  if (!out.hasOneUse()) {
    IVLOG(3, "cacheReduce failed: multiple uses");
    return None;
  }
  auto &finalUse = *out.use_begin();

  // Prep for generation
  auto loc = reduce.getLoc();
  auto builder = OpBuilder::atBlockBegin(par.getBody());

  // Allocate a temporary buffer
  auto eltType = reduce.getMemRefType().getElementType();
  auto type = MemRefType::get(rap.innerCount, eltType);
  auto localBuf = builder.create<AllocOp>(loc, type);

  // If it's not an assign, clear it to the reduction identity
  Value initBuf = localBuf;
  if (reduce.agg() != AtomicRMWKind::assign) {
    auto ident = createIdentity(builder, loc, reduce.agg(), eltType);
    initBuf = createInitLoop(builder, loc, localBuf, ident);
  }

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
  finalUse.set(copyLoop.getResult(0));
  reduce.erase();

  return CacheInfo{
      /*copyLoopOp=*/copyLoop,
      /*relativeAccess=*/*maybeRap,
  };
}

} // namespace pmlc::dialect::pxa
