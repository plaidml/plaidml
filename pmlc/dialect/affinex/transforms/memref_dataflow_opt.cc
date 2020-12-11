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

    if (LHS.map != RHS.map) {
      return false;
    }

    if (LHS.indices.size() != RHS.indices.size()) {
      return false;
    }

    for (size_t i = 0; i < LHS.indices.size(); ++i) {
      if (LHS.indices[i] != RHS.indices[i]) {
        return false;
      }
    }
    return true;
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
      // we only care about stores and loads
      if (auto storeOp = dyn_cast<AffineWriteOpInterface>(op)) {
        MemAccess access{storeOp.getMemRef(), storeOp.getAffineMap(),
                         storeOp.getMapOperands()};
        // if we have already stored to this location in this block
        if (lastStoreOps.find(access) != lastStoreOps.end()) {
          auto lastStoreOp = lastStoreOps[access];
          // erase the previous store (later)
          opsToErase.push_back(lastStoreOp);
        }
        // update last store op
        lastStoreOps[access] = storeOp;
      } else if (auto loadOp = dyn_cast<AffineReadOpInterface>(op)) {
        MemAccess access{loadOp.getMemRef(), loadOp.getAffineMap(),
                         loadOp.getMapOperands()};
        // if we have already stored to this location in this block
        if (lastStoreOps.find(access) != lastStoreOps.end()) {
          auto lastStoreOp = lastStoreOps[access];
          // replace all uses of the load with the store value
          loadOp.getValue().replaceAllUsesWith(lastStoreOp.getValueToStore());
          // erase the load (later)
          opsToErase.push_back(loadOp);
          // consider the memref for deletion (later)
          memrefsToErase.insert(loadOp.getMemRef());
        }
      }
    }

    // erase all redundant loads and stores
    for (auto *op : opsToErase) {
      op->erase();
    }

    // BEGIN leverage from upstream MLIR
    // Check if the store fwd'ed memrefs are now left with only stores and can
    // thus be completely deleted. Note: the canonicalize pass should be able
    // to do this as well, but we'll do it here since we collected these anyway.
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
    // END leverage from upstream MLIR
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
