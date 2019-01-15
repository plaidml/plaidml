// Copyright 2018, Intel Corporation

#include "tile/codegen/unroll.h"

#include "base/util/logging.h"
#include "base/util/throw.h"
#include "tile/codegen/tags.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

namespace {

struct IndexValue {
  const Index* idx;
  size_t value;
};

using EnumerateFunc = std::function<void(const std::vector<IndexValue>& idxs)>;

void EnumerateIndexes(const std::vector<IndexValue>& idxs, size_t idx_num, const EnumerateFunc& func) {
  if (idx_num == idxs.size()) {
    func(idxs);
  } else {
    for (size_t i = 0; i < idxs[idx_num].idx->range; i++) {
      auto next = idxs;
      next[idx_num].value = i;
      EnumerateIndexes(next, idx_num + 1, func);
    }
  }
}

void EvalWithValues(Block* block, const std::map<std::string, int64_t>& fixed) {
  IVLOG(3, "EvalWithValues> block: " << block->name << ", fixed: " << fixed);
  block->location.unit = block->location.unit.partial_eval(fixed);
  for (auto& ref : block->refs) {
    ref.location.unit = ref.location.unit.partial_eval(fixed);
    for (auto& aff : ref.access) {
      aff = aff.partial_eval(fixed);
    }
    if (ref.cache_unit) {
      ref.cache_unit = ref.cache_unit->partial_eval(fixed);
    }
  }
  for (auto& constraint : block->constraints) {
    constraint = constraint.partial_eval(fixed);
  }
  for (const auto& stmt : block->stmts) {
    auto inner = Block::Downcast(stmt);
    if (inner) {
      for (auto& idx : inner->idxs) {
        idx.affine = idx.affine.partial_eval(fixed);
      }
      std::map<std::string, int64_t> next;
      for (const auto& item : fixed) {
        if (item.first[0] == '#') {
          next.emplace(item.first, item.second);
        }
      }
      std::vector<std::string> prune_idxs;
      for (const auto& idx : inner->idxs) {
        if (idx.range == 1 && idx.affine.isConstant()) {
          prune_idxs.push_back(idx.name);
          next.emplace(idx.name, idx.affine.constant());
        }
      }
      EvalWithValues(inner.get(), next);
      for (const auto& idx_name : prune_idxs) {
        auto it = std::find_if(inner->idxs.begin(), inner->idxs.end(),
                               [&idx_name](const Index& idx) { return idx.name == idx_name; });
        inner->idxs.erase(it);
      }
    }
  }
}

void EvalInner(Block* outer,                         //
               Block* block,                         //
               const std::vector<IndexValue>& idxs,  //
               const proto::UnrollPass& options) {
  IVLOG(3, "EvalInner> " << outer->name);
  std::map<std::string, int64_t> fixed;
  size_t last_stride = 1;
  size_t expand_val = 0;
  for (const auto& idx_val : idxs) {
    fixed.emplace(idx_val.idx->name, idx_val.value);
    expand_val += last_stride * idx_val.value;
    last_stride *= idx_val.idx->range;
  }
  std::string part_suffix;
  if (!options.part_name().empty()) {
    part_suffix = str(boost::format("%%%1%_%2%") % options.part_name() % expand_val);
  }
  if (!options.expand_idx().empty()) {
    fixed.emplace(options.expand_idx(), expand_val);
  }
  EvalWithValues(block, fixed);
  for (const auto& stmt : block->stmts) {
    auto inner = Block::Downcast(stmt);
    if (inner) {
      if (!part_suffix.empty()) {
        inner->name += part_suffix;
      }
      for (auto& ref : inner->refs) {
        if (!ref.from.empty()) {
          auto block_ref = block->ref_by_into(ref.from);
          for (size_t i = 0; i < ref.access.size(); i++) {
            ref.access[i] += block_ref->access[i];
          }
          if (options.make_views()) {
            auto outer_ref = outer->ref_by_into(block_ref->from);
            if (!(block_ref->interior_shape == outer_ref->interior_shape)) {
              auto view = *block_ref;
              view.from = outer_ref->from;
              view.into = block_ref->into + part_suffix;
              IVLOG(3, "  make view: " << view.into << " from: " << view.from << " via: " << block_ref->from);
              if (ref.cache_unit || outer_ref->cache_unit) {
                IVLOG(3, "  with cache: " << *outer_ref->cache_unit);
                view.cache_unit = outer_ref->cache_unit;
              }
              for (size_t i = 0; i < ref.access.size(); i++) {
                auto const_access = block_ref->access[i].constant();
                view.access[i] = outer_ref->access[i] + const_access;
                ref.access[i] -= const_access;
              }
              ref.from = view.into;
              IVLOG(2, "view: " << view);
              outer->refs.emplace_back(view);
            }
          }
        }
      }
    }
  }
}

void UnrollBlock(Block* outer,                //
                 Block* block,                //
                 const StatementIt& it_stmt,  //
                 const Tags& reqs,            //
                 const proto::UnrollPass& options) {
  if (block->has_tags(reqs)) {
    std::vector<IndexValue> idxs;
    idxs.reserve(block->idxs.size());
    for (const auto& idx : block->idxs) {
      idxs.emplace_back(IndexValue{&idx, 0});
    }
    EnumerateIndexes(idxs, 0, [&](const std::vector<IndexValue>& idxs) {
      auto clone = CloneBlock(*block);
      EvalInner(outer, clone.get(), idxs, options);
      for (const auto& stmt : clone->stmts) {
        outer->stmts.insert(it_stmt, stmt);
      }
    });
    outer->stmts.erase(it_stmt);
  } else {
    PreIterate(block, [&](const StatementIt& it) {
      auto inner = Block::Downcast(*it);
      if (inner) {
        UnrollBlock(block, inner.get(), it, reqs, options);
      }
    });
  }
}

}  // namespace

void UnrollPass(Block* root, const proto::UnrollPass& options) {
  auto reqs = FromProto(options.reqs());
  PreIterate(root, [&](const StatementIt& it) {
    auto inner = Block::Downcast(*it);
    if (inner) {
      UnrollBlock(root, inner.get(), it, reqs, options);
    }
  });
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
