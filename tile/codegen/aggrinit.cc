// Copyright 2018, Intel Corporation

#include <vector>

#include "tile/codegen/aggrinit.h"
#include "tile/codegen/deps.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT
using namespace math;    // NOLINT

void AggregationBlockOutputInitialization(const stripe::Block* const block,
                                          const AggregationBlockOutputInitializationPass* aggregationPass) {
  if (block == nullptr) {
    throw new std::runtime_error("AggregationBlockOutputInitialization' block parameter cannot be nullptr.");
  }

  // Handle local buffers.
  // When two same refinements are used in the same stripe block,
  // do not emit duplicated local declarations.
  std::set<std::string> dup_ref;

  // Now, add any new locals
  for (const auto& ref : block->refs) {
    if (ref.dir == stripe::RefDir::None && dup_ref.find(ref.into()) == dup_ref.end() && !ref.has_tag("user")) {
      const_cast<AggregationBlockOutputInitializationState&>(aggregationPass->state)
          .blocksWithInits.emplace_back(
              AggregationBlockOutputInitializationNode(const_cast<stripe::Block*>(block), &ref));
    }
  }

  const Block* prevBlock = aggregationPass->state.prevBlock;
  if (prevBlock == nullptr) {
    // Nothing to do here.
    return;
  }

  std::vector<const Refinement*> outRefs = block->ref_outs(true);
  // We should handle only when there is one output (aggregation operation).
  if (outRefs.size() != 1) {
    return;
  }

  const Refinement* dest = outRefs[0];
  std::string aggOp = dest->agg_op;
  if (aggOp != "add" && aggOp != "mul" && aggOp != "max" && aggOp != "min") {
    return;
  }

  auto prevRefIter = prevBlock->ref_by_into(dest->from);
  if (prevRefIter == prevBlock->refs.end()) {
    throw std::runtime_error(
        "AggregationBlockOutputInitializationPass: Didn't find referenced Refinement from outer block.");
  }

  if (prevRefIter->agg_op == "" || prevRefIter->agg_op == "assign") {
    const_cast<AggregationBlockOutputInitializationState&>(aggregationPass->state)
        .blocksWithInits.emplace_back(
            AggregationBlockOutputInitializationNode(const_cast<stripe::Block*>(block), dest));
  } else {
    if (prevRefIter->agg_op != aggOp) {
      // TODO: Create a temp buffer here.
      throw std::runtime_error("Nested agg ops of different type is not supported at the moment");
    }
  }
}

void AggregationBlockOutputInitializationPass::Apply(CompilerState* state) const {
  auto reqs = stripe::FromProto(options_.reqs());
  RunOnBlocksRecurse(state->entry(), reqs, this);

  // Generate the specials AggInit Calls
  for (const auto& toInit : this->state.blocksWithInits) {
    assert(toInit.blockToAddTo != nullptr);
    assert(toInit.refToInitialize != nullptr);
    auto aggInit = std::make_shared<Special>();
    aggInit->name = "agg_init";
    aggInit->outputs.emplace_back(toInit.refToInitialize->into());
    toInit.blockToAddTo->stmts.emplace_front(aggInit);
  }

  const_cast<AggregationBlockOutputInitializationState&>(this->state).Clear();
}

void AggregationBlockOutputInitializationPass::AddRefinementToInit(stripe::Block* block,
                                                                   const stripe::Refinement* ref) {
  state.blocksWithInits.push_back(AggregationBlockOutputInitializationNode((block), ref));
}

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<AggregationBlockOutputInitializationPass,
                     proto::AggregationBlockOutputInitializationPass>::Register();
  return 0;
}();
}  // namespace
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
