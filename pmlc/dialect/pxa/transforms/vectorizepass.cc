// Copyright 2020 Intel Corporation

#include "pmlc/dialect/pxa/transforms/vectorize.h"

#include "pmlc/dialect/pxa/transforms/pass_detail.h"

#include "pmlc/util/logging.h"

#define PXA_EXAMPLE_VECTOR_WIDTH_IN_BYTES 32
#define PXA_EXAMPLE_ELEM_MIN_WIDTH 1

namespace pmlc::dialect::pxa {

struct VectorizeExample : public VectorizeExampleBase<VectorizeExample> {
  void runOnFunction() final {
    auto func = getFunction();
    // Autotile only the outermost loops
    for (auto &op : func.getBody().front()) {
      auto loop = mlir::dyn_cast<mlir::AffineParallelOp>(op);
      if (!loop) {
        continue;
      }
      //   auto ranges = loop.getConstantRanges();
      //   if (!ranges) {
      //     return;
      //   }

      bool vectorized = false;
      for (unsigned int i = 0; i < loop.getIVs().size(); i++) {
        auto blockArg = loop.getIVs()[i];
        IVLOG(1, "Lubo3 VectorizePass!!!" << vectorized << ":"
                                          << blockArg.getArgNumber());
        vectorized |= performVectorization(loop, blockArg,
                                           PXA_EXAMPLE_VECTOR_WIDTH_IN_BYTES,
                                           PXA_EXAMPLE_ELEM_MIN_WIDTH);
        // IVLOG(1, "Lubo VectorizePass!!!" << vectorized << ":" <<
        // blockArg.getArgNumber());
        if (vectorized) {
          // IVLOG(1, "Lubo2 VectorizePass!!!" << vectorized << ":" <<
          // blockArg.getArgNumber());
          return;
        }
      }

      IVLOG(4,
            (vectorized
                 ? "Vectorization: Performed"
                 : "Cannot Vectorize: No instruction vectorized successfully"));
    }
  }
};

std::unique_ptr<mlir::Pass> createVectorizeExamplePass() {
  return std::make_unique<VectorizeExample>();
}

} // namespace pmlc::dialect::pxa
