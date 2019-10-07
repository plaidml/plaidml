// Copyright 2018, Intel Corporation

#include <memory>
#include <set>
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

  // Now, add any new locals that have agg_op.
  for (const auto& ref : block->refs) {
    if (ref.dir == stripe::RefDir::None && dup_ref.find(ref.into()) == dup_ref.end() && !ref.has_tag("user")) {
      std::string aggOp = ref.agg_op;
      if (!aggOp.empty()) {
        const_cast<AggregationBlockOutputInitializationPass*>(aggregationPass)
            ->AddRefinementToInit(const_cast<stripe::Block*>(block), &ref, nullptr, aggOp);
      }
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
    const_cast<AggregationBlockOutputInitializationPass*>(aggregationPass)
        ->AddRefinementToInit(const_cast<stripe::Block*>(prevBlock), &(*prevRefIter), block, aggOp);
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
    assert(toInit.initType == AggregationInitType::ADD || toInit.initType == AggregationInitType::MUL ||
           toInit.initType == AggregationInitType::MIN || toInit.initType == AggregationInitType::MAX);
    auto aggInit = std::make_shared<Special>();
    std::string aggInitName = "";
    switch (toInit.initType) {
      case AggregationInitType::ADD:
        aggInitName = "agg_init_add";
        break;
      case AggregationInitType::MUL:
        aggInitName = "agg_init_mul";
        break;
      case AggregationInitType::MIN:
        aggInitName = "agg_init_min";
        break;
      case AggregationInitType::MAX:
        aggInitName = "agg_init_max";
        break;
      default:
        throw std::runtime_error("Invalid Aggregation Initialization Type value.");
    }
    aggInit->name = aggInitName;
    aggInit->outputs.emplace_back(toInit.refToInitialize->into());
    if (toInit.statementToAddBefore == nullptr) {
      toInit.blockToAddTo->stmts.emplace_front(aggInit);
    } else {
      // Insert the element before the toInit.statementToAddBefore element.
      auto it = toInit.blockToAddTo->stmts.begin();
      auto end = toInit.blockToAddTo->stmts.end();
      for (; it != end; ++it) {
        if (it->get() == toInit.statementToAddBefore) {
          break;
        }
      }

      if (it == end) {
        throw std::runtime_error("The toInit.statementToAddBefore must be in the list.");
      }

      toInit.blockToAddTo->stmts.insert(it, aggInit);
    }
  }

  const_cast<AggregationBlockOutputInitializationState&>(this->state).Clear();
}

void AggregationBlockOutputInitializationPass::AddRefinementToInit(stripe::Block* toBlock,
                                                                   const stripe::Refinement* ref,
                                                                   const stripe::Statement* beforeStatement,
                                                                   const std::string initTypeStr) {
  AggregationInitType initType = AggregationInitType::NONE;
  if (initTypeStr == "add") {
    initType = AggregationInitType::ADD;
  } else if (initTypeStr == "mul") {
    initType = AggregationInitType::MUL;
  } else if (initTypeStr == "min") {
    initType = AggregationInitType::MIN;
  } else if (initTypeStr == "max") {
    initType = AggregationInitType::MAX;
  } else {
    throw std::runtime_error("Valid initTypeStr must be specified for AggregationInitType.");
  }

  state.blocksWithInits.emplace_back(AggregationBlockOutputInitializationNode(toBlock, ref, beforeStatement, initType));
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
