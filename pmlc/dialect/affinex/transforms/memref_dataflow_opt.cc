// Copyright 2020 Intel Corporation

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "pmlc/dialect/affinex/transforms/pass_detail.h"

using namespace mlir; // NOLINT

struct MemAccess {
  Value memref;
  AffineMap map;
  llvm::SmallVector<Value, 4> indices;
  MemAccess() = default;
};

namespace llvm {
template <>
struct DenseMapInfo<MemAccess> {
  static inline MemAccess getEmptyKey() {
    MemAccess access;
    access.memref = DenseMapInfo<mlir::Value>::getEmptyKey();
    return access;
  }
  static inline MemAccess getTombstoneKey() {
    MemAccess access;
    access.memref = DenseMapInfo<mlir::Value>::getEmptyKey();
    return access;
  }
  static unsigned getHashValue(const MemAccess &Val) {
    return hash_combine(
        Val.memref, Val.map,
        hash_combine_range(Val.indices.begin(), Val.indices.end()));
  }
  static bool isEqual(const MemAccess &LHS, const MemAccess &RHS) {
    if (LHS.memref != RHS.memref) {
      return false;
    }
    return true;

    /*
    AffineValueMap lhsValueMap, rhsValueMap, diff;
    lhsValueMap.reset(LHS.map, LHS.indices);
    rhsValueMap.reset(RHS.map, RHS.indices);
    AffineValueMap::difference(lhsValueMap, rhsValueMap, &diff);
    return llvm::all_of(diff.getAffineMap().getResults(),
                        [](AffineExpr e) { return e == 0; });
    */
  }
};
} // namespace llvm

namespace pmlc::dialect::affinex {

struct AffinexMemRefDataFlowOpt
    : public AffinexMemRefDataFlowOptBase<AffinexMemRefDataFlowOpt> {

  void memref_dataflow(Block &block) {
    llvm::SmallPtrSet<Value, 4> memrefsToErase;
    llvm::SmallVector<Operation *, 8> opsToErase;
    llvm::DenseMap<MemAccess, AffineWriteOpInterface> lastStoreOps;

    for (Operation &op : block.getOperations()) {
      if (auto storeOp = dyn_cast<AffineWriteOpInterface>(op)) {
        MemAccess access;
        access.memref = storeOp.getMemRef();
        access.map = storeOp.getAffineMap();
        access.indices.append(storeOp.getMapOperands().begin(),
                              storeOp.getMapOperands().end());

        if (lastStoreOps.find(access) != lastStoreOps.end()) {
          auto lastStoreOp = lastStoreOps[access];
          opsToErase.push_back(lastStoreOp);
        }

        lastStoreOps[access] = storeOp;
      } else if (auto loadOp = dyn_cast<AffineReadOpInterface>(op)) {
        MemAccess access;
        access.memref = loadOp.getMemRef();
        access.map = loadOp.getAffineMap();
        access.indices.append(loadOp.getMapOperands().begin(),
                              loadOp.getMapOperands().end());

        if (lastStoreOps.find(access) != lastStoreOps.end()) {
          auto lastStoreOp = lastStoreOps[access];
          loadOp.getValue().replaceAllUsesWith(lastStoreOp.getValueToStore());
          memrefsToErase.insert(loadOp.getMemRef());
          opsToErase.push_back(loadOp);
        }
      }
    }

    for (auto *op : opsToErase)
      op->erase();

    for (auto memref : memrefsToErase) {
      // If the memref hasn't been alloc'ed in this function, skip.
      Operation *defOp = memref.getDefiningOp();
      if (!defOp || !isa<AllocOp>(defOp))
        // TODO: if the memref was returned by a 'call' operation, we
        // could still erase it if the call had no side-effects.
        continue;
      if (llvm::any_of(memref.getUsers(), [&](Operation *ownerOp) {
            return !isa<AffineWriteOpInterface, DeallocOp>(ownerOp);
          }))
        continue;

      // Erase all stores, the dealloc, and the alloc on the memref.
      for (auto *user : llvm::make_early_inc_range(memref.getUsers()))
        user->erase();
      defOp->erase();
    }
  }

  void runOnFunction() override {
    mlir::Region &region = getFunction().getRegion();
    for (mlir::Block &block : region.getBlocks()) {
      memref_dataflow(block);
    }
  }
};

std::unique_ptr<mlir::Pass> createAffinexMemRefDataFlowOpt() {
  return std::make_unique<AffinexMemRefDataFlowOpt>();
}
} // namespace pmlc::dialect::affinex
