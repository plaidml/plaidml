// Copyright 2019, Intel Corporation

#pragma once

#include <vector>

#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

// Forward declarations
class AggregationBlockOutputInitializationPass;
void RunOnBlocksRecurse(stripe::Block* block, const stripe::Tags& reqs,
                        const AggregationBlockOutputInitializationPass* aggregationPass);
void AggregationBlockOutputInitialization(const stripe::Block* const block,
                                          const AggregationBlockOutputInitializationPass* aggregationPass);

class AggregationBlockOutputInitializationNode final {
 public:
  explicit AggregationBlockOutputInitializationNode(stripe::Block* blockToAddToIn,
                                                    const stripe::Refinement* refToInitializeIn)
      : blockToAddTo(blockToAddToIn), refToInitialize(refToInitializeIn) {}

  stripe::Block* blockToAddTo;
  const stripe::Refinement* refToInitialize;

  AggregationBlockOutputInitializationNode() = delete;
};

class AggregationBlockOutputInitializationState final {
 public:
  AggregationBlockOutputInitializationState() : prevBlock(nullptr) {}
  stripe::Block* prevBlock;
  std::vector<AggregationBlockOutputInitializationNode> blocksWithInits;

  void Clear() {
    prevBlock = nullptr;
    blocksWithInits.clear();
  }
};

class AggregationBlockOutputInitializationPass final : public CompilePass {
 public:
  explicit AggregationBlockOutputInitializationPass(const proto::AggregationBlockOutputInitializationPass& options)
      : options_{options} {}
  void Apply(CompilerState* state) const final;
  void AddRefinementToInit(stripe::Block* block, const stripe::Refinement* ref);

  AggregationBlockOutputInitializationState state;

 private:
  proto::AggregationBlockOutputInitializationPass options_;
};

// Traverse blocks
void RunOnBlocksRecurse(stripe::Block* block, const stripe::Tags& reqs,
                        const AggregationBlockOutputInitializationPass* aggregationPass) {
  for (auto stmt_it = block->stmts.rbegin(); stmt_it != block->stmts.rend(); ++stmt_it) {
    auto inner = stripe::Block::Downcast(*stmt_it);
    if (inner) {
      stripe::Block* const innerBlock = inner.get();
      AggregationBlockOutputInitialization(innerBlock, aggregationPass);
      AggregationBlockOutputInitializationState& state =
          const_cast<AggregationBlockOutputInitializationState&>(aggregationPass->state);
      stripe::Block* pBlock = state.prevBlock;
      state.prevBlock = innerBlock;
      RunOnBlocksRecurse(innerBlock, reqs, aggregationPass);
      state.prevBlock = pBlock;
    }
  }
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
