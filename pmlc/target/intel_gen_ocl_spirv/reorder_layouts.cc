// Copyright 2020, Intel Corporation

#include "pmlc/dialect/pxa/transforms/reorder_layouts.h"
#include "mlir/Support/DebugStringHelper.h"
#include "pmlc/dialect/pxa/transforms/gpu_thread.h"
#include "pmlc/target/intel_gen_ocl_spirv/pass_detail.h"
#include "pmlc/target/intel_gen_ocl_spirv/passes.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/tags.h"

#include "mlir/Pass/Pass.h"

namespace pmlc::target::intel_gen_ocl_spirv {
namespace pxa = pmlc::dialect::pxa;

namespace {

/// Function modeling loop scheduling on Intel GPU's.
/// In Intel's OpenCL implementation global and local dimensions
/// are scanned from first to last which is opposite to order
/// used for normal loops.
/// Additionally it interleaves trully parallel loops with
/// inner ones, which will be lowered as sequential.
/// If sub-group size is not defined (i.e. 1 or empty) it will
/// start with parallel loop to help backend in preserving
/// data locality when sub-grouping.
/// It assumes that difference of one in OpenCL schedule dimensions
/// is equal to one iteration of sequential loop.
pxa::LoopNestSchedule
intelGenOclScheduleModel(mlir::ArrayRef<mlir::AffineParallelOp> loopNest) {
  pxa::LoopNestSchedule result;
  result.resize(loopNest.size());
  unsigned subGroupSize = getIntegerTag(loopNest[0], subgroupSizeTag(), 1);
  unsigned parallelStride = subGroupSize;
  // Walk local and global dimensions with operands from left to right.
  for (unsigned level = 2; level > 0; --level) {
    mlir::AffineParallelOp local = loopNest[level - 1];
    pxa::OperandSchedule &operandSched = result[level - 1];
    auto optRanges = local.getConstantRanges();
    if (!optRanges.hasValue())
      return pxa::naiveScheduleModel(loopNest);

    for (int64_t range : optRanges.getValue()) {
      operandSched.push_back(parallelStride);
      parallelStride *= range;
    }
  }
  int64_t sequenceStride = 1;
  // For undefined sub-group assume biggest one will be selected - 32.
  if (subGroupSize == 1)
    sequenceStride = 32;
  // Next walk non-parallel dimensions with operands from right to left.
  for (unsigned level = loopNest.size(); level > 2; --level) {
    mlir::AffineParallelOp sequence = loopNest[level - 1];
    pxa::OperandSchedule &operandSched = result[level - 1];
    operandSched.resize(sequence.getIVs().size());
    auto optRanges = sequence.getConstantRanges();
    if (!optRanges.hasValue())
      return pxa::naiveScheduleModel(loopNest);
    for (unsigned argIdx = operandSched.size(); argIdx > 0; --argIdx) {
      operandSched[argIdx - 1] = sequenceStride;
      sequenceStride *= optRanges.getValue()[argIdx - 1];
    }
  }
  return result;
}

/// ReorderCreator implemented as function object.
/// In addition to creating reorder it also aligns it with GPU threading
/// semantics - two level global & local parallel.
struct ThreadedReorderCreator {
  explicit ThreadedReorderCreator(unsigned maxThreads)
      : maxThreads(maxThreads) {}

  mlir::Value operator()(mlir::Location loc, mlir::OpBuilder &builder,
                         pxa::ReorderDesc &reorderDesc, mlir::Value srcMem) {
    mlir::Value newMem = pxa::createReorder(loc, builder, reorderDesc, srcMem);
    auto reorderParallel = newMem.getDefiningOp<mlir::AffineParallelOp>();
    pxa::gpuThreadParallelOp(maxThreads, reorderParallel);
    return newMem;
  }

  unsigned maxThreads;
};

class IntelGenOclReorderLayoutsPass final
    : public IntelGenOclReorderLayoutsBase<IntelGenOclReorderLayoutsPass> {
public:
  IntelGenOclReorderLayoutsPass() = default;

  IntelGenOclReorderLayoutsPass(unsigned maxThreads, bool allowReorder) {
    this->maxThreads = maxThreads;
    this->allowReorder = allowReorder;
  }

  void runOnFunction() {
    mlir::FuncOp func = getFunction();
    mlir::DenseMap<mlir::Value, pxa::MemoryUsageDesc> globalMemory =
        pxa::gatherGlobalMemoryDescs(func, intelGenOclScheduleModel);
    for (auto &valueDesc : globalMemory) {
      pxa::MemoryUsageDesc &memoryDesc = valueDesc.second;
      IVLOG(3, "Optimizing layout for " << mlir::debugString(memoryDesc.value));
      mlir::Optional<pxa::ReorderDesc> optReorder =
          pxa::optimizeLayoutForReads(memoryDesc);
      if (!optReorder.hasValue()) {
        IVLOG(3, "Could not select more optimal layout");
        continue;
      }
      pxa::ReorderDesc &reorder = optReorder.getValue();
      IVLOG(3, "Optimized layout: " << mlir::debugString(reorder.reorderMap));
      if (mlir::succeeded(pxa::convertMemoryLayout(memoryDesc.value, reorder)))
        continue;
      if (!allowReorder) {
        IVLOG(3,
              "Failed to change layout in-place, separate reorder not allowed");
        continue;
      }
      IVLOG(3, "Failed to change layout in-place, inserting reorder");
      pxa::reorderMemoryReads(ThreadedReorderCreator(maxThreads), reorder,
                              memoryDesc);
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createIntelGenOclReorderLayoutsPass() {
  return std::make_unique<IntelGenOclReorderLayoutsPass>();
}

std::unique_ptr<mlir::Pass>
createIntelGenOclReorderLayoutsPass(unsigned maxThreads, bool allowReorder) {
  return std::make_unique<IntelGenOclReorderLayoutsPass>(maxThreads,
                                                         allowReorder);
}

} // namespace pmlc::target::intel_gen_ocl_spirv
