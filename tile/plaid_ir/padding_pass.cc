// Copyright 2019, Intel Corporation

#include "tile/plaid_ir/padding_pass.h"

#include <iostream>

#include "tile/plaid_ir/analysis.h"
#include "tile/plaid_ir/ops.h"

namespace vertexai {
namespace tile {
namespace plaid_ir {

// Compute for a given tensor the range of each of it's dimensions
std::vector<AffineRange> ComputeUnboundedRanges(Value* val) {
  // Initialize ranges to 'no-use' case
  std::vector<AffineRange> r;
  // Go over all uses
  for (const auto& use : val->getUses()) {
    std::vector<AffineRange> inner;
    if (auto ref = mlir::dyn_cast<RefineOp>(use.getOwner())) {
      // If it's a refinement, recurse
      inner = ComputeUnboundedRanges(ref.result());
      // In the refinement itself has no uses, ignore
      if (inner.size() == 0) {
        continue;
      }
      assert(ref.offsets().end() - ref.offsets().begin() == inner.size());
      // Add the effect of the offset to the inner ranges
      for (size_t i = 0; i < inner.size(); i++) {
        inner[i] += UnboundedRange(*(ref.offsets().begin() + i));
      }
    } else if (auto op = mlir::dyn_cast<LoadOp>(use.getOwner())) {
      inner.resize(op.from()->getType().cast<TensorType>().dims().size());
    } else if (auto op = mlir::dyn_cast<StoreOp>(use.getOwner())) {
      inner.resize(op.into()->getType().cast<TensorType>().dims().size());
    } else {
      throw std::runtime_error("Invalid type");
    }
    if (r.size() == 0) {
      // If this is the first use, set
      r = inner;
    } else {
      // Otherwise, union in
      assert(r.size() == inner.size());
      for (size_t i = 0; i < r.size(); i++) {
        r[i].merge(inner[i]);
      }
    }
  }
  return r;
}

void PaddingPass::runOnFunction() {
  mlir::Function& f = getFunction();
  // Get the unbounded access ranges for each function input
  std::cout << "Args\n";
  for (const auto& arg : f.getArguments()) {
    std::vector<AffineRange> final = ComputeUnboundedRanges(arg);
    for (const auto& range : final) {
      std::cout << range.min << ":" << range.max << " ";
    }
    std::cout << "\n";
  }
  std::cout << "Temps\n";
  // Get the unbounded access range of each allocation
  f.walk<AllocateOp>([](AllocateOp op) {
    std::vector<AffineRange> final = ComputeUnboundedRanges(op.res());
    for (const auto& range : final) {
      std::cout << range.min << ":" << range.max << " ";
    }
    std::cout << "\n";
  });
}

}  // namespace plaid_ir
}  // namespace tile
}  // namespace vertexai
