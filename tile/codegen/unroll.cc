// Copyright 2018, Intel Corporation

#include "tile/codegen/unroll.h"

#include "base/util/logging.h"
#include "tile/codegen/tags.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

static void DeepUnroll(StatementList* list, StatementIt it, const Block& base, std::map<std::string, int64_t>* fixed,
                       size_t idx_num, int64_t mul) {
  if (idx_num == base.idxs.size()) {
    auto clone = CloneBlock(base);
    for (auto& ref : clone->refs) {
      for (auto& aff : ref.access) {
        aff = aff.eval(*fixed);
      }
    }
    int64_t loc = 0;
    for (const auto& idx : base.idxs) {
      loc *= idx.range;
      loc += fixed->at(idx.name);
    }
    loc *= mul;
    // Copy across inner statements
    for (const auto& stmt : clone->stmts) {
      auto inner = Block::Downcast(stmt);
      if (!inner) {
        throw std::runtime_error("Unhandled statment type in DeepUnroll");
      }
      inner->location.unit += loc;
      for (auto& idx : inner->idxs) {
        idx.affine = idx.affine.eval(*fixed);
      }
      for (auto& ref : inner->refs) {
        if (ref.dir == RefDir::None) {
          continue;
        }
        const auto& oref = clone->ref_by_into(ref.from);
        ref.from = oref->from;
        for (size_t i = 0; i < ref.access.size(); i++) {
          ref.access[i] += oref->access[i];
        }
      }
      list->insert(it, inner);
    }
  } else {
    for (int64_t i = 0; i < static_cast<int64_t>(base.idxs[idx_num].range); i++) {
      (*fixed)[base.idxs[idx_num].name] = i;
      DeepUnroll(list, it, base, fixed, idx_num + 1, mul);
    }
  }
}

static void UnrollTags(StatementList* list, StatementIt it, stripe::Block* block, const Tags& reqs, int64_t mul) {
  if (HasTags(*block, reqs)) {
    std::map<std::string, int64_t> fixed;
    DeepUnroll(list, it, *block, &fixed, 0, mul);
    list->erase(it);
  } else {
    auto iit = block->stmts.begin();
    while (iit != block->stmts.end()) {
      auto next = iit;
      ++next;
      auto inner = stripe::Block::Downcast(*iit);
      if (inner) {
        UnrollTags(&block->stmts, iit, inner.get(), reqs, mul);
      }
      iit = next;
    }
  }
}

void UnrollPass(Block* root, const proto::UnrollPass& options) {
  auto reqs = FromProto(options.reqs());
  auto it = root->stmts.begin();
  while (it != root->stmts.end()) {
    auto next = it;
    ++next;
    auto inner = stripe::Block::Downcast(*it);
    if (inner) {
      UnrollTags(&root->stmts, it, inner.get(), reqs, options.loc_mul());
    }
    it = next;
  }
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
