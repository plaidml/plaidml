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

void EvalWithValues(Block* block, const std::map<std::string, int64_t>& fixed, bool top) {
  IVLOG(3, "EvalWithValues> block: " << block->name << ", fixed: " << fixed);
  block->location.unit = block->location.unit.partial_eval(fixed);
  for (auto& ref : block->refs) {
    ref.location.unit = ref.location.unit.partial_eval(fixed);
    for (auto& aff : ref.access) {
      aff = aff.partial_eval(fixed);
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
      if (top) {
        for (auto& ref : inner->refs) {
          if (!ref.from.empty()) {
            auto from = ref.from;
            if (ref.bank_dim) {  // HACK: can we make this configurable somehow?
              from = ref.bank_dim->orig_name;
            }
            auto outer_ref = block->ref_by_into(from);
            for (size_t i = 0; i < ref.access.size(); i++) {
              ref.access[i] += outer_ref->access[i];
            }
          }
        }
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
      EvalWithValues(inner.get(), next, false);
      for (const auto& idx_name : prune_idxs) {
        auto it = std::find_if(inner->idxs.begin(), inner->idxs.end(),
                               [&idx_name](const Index& idx) { return idx.name == idx_name; });
        inner->idxs.erase(it);
      }
    }
  }
}

void EvalInner(Block* block, const std::vector<IndexValue>& idxs, const std::string& expand_idx) {
  std::map<std::string, int64_t> fixed;
  size_t last_stride = 1;
  size_t complete_value = 0;
  for (const auto& item : idxs) {
    fixed.emplace(item.idx->name, item.value);
    complete_value += last_stride * item.value;
    last_stride *= item.idx->range;
  }
  if (!expand_idx.empty()) {
    fixed.emplace(expand_idx, complete_value);
  }
  EvalWithValues(block, fixed, true);
}

void PreIterate(Block* block, std::function<void(const StatementIt& it)> func) {
  auto it = block->stmts.begin();
  while (it != block->stmts.end()) {
    auto next = it;
    ++next;
    func(it);
    it = next;
  }
}

void UnrollBlock(Block* outer, Block* block,  //
                 const StatementIt& it_stmt,  //
                 const Tags& reqs,            //
                 const std::string& expand_idx) {
  if (block->has_tags(reqs)) {
    std::vector<IndexValue> idxs;
    for (const auto& idx : block->idxs) {
      idxs.emplace_back(IndexValue{&idx, 0});
    }
    EnumerateIndexes(idxs, 0, [&](const std::vector<IndexValue>& idxs) {
      auto clone = CloneBlock(*block);
      EvalInner(clone.get(), idxs, expand_idx);
      for (const auto& stmt : clone->stmts) {
        outer->stmts.insert(it_stmt, stmt);
      }
    });
    outer->stmts.erase(it_stmt);
  } else {
    PreIterate(block, [&](const StatementIt& it) {
      auto inner = Block::Downcast(*it);
      if (inner) {
        UnrollBlock(block, inner.get(), it, reqs, expand_idx);
      }
    });
  }
}

std::vector<Index>::const_iterator GetIndexWithTag(const Block& block, const std::string& tag) {
  auto it_end = block.idxs.end();
  for (auto it = block.idxs.begin(); it != it_end; it++) {
    if (it->tags.count(tag)) {
      return it;
    }
  }
  return it_end;
}

void EvalIndex(Block* block, const std::string& tag, int64_t value) {
  auto tagged_name = str(boost::format("#%1%") % tag);
  std::map<std::string, int64_t> fixed = {
      {tagged_name, value},
  };

  auto it_idx = GetIndexWithTag(*block, tag);
  if (it_idx != block->idxs.end()) {
    fixed.emplace(it_idx->name, value);
  }

  block->location.unit = block->location.unit.partial_eval(fixed);
  for (auto& ref : block->refs) {
    ref.location.unit = ref.location.unit.partial_eval(fixed);
    for (auto& aff : ref.access) {
      aff = aff.partial_eval(fixed);
    }
  }
  for (auto& constraint : block->constraints) {
    constraint = constraint.partial_eval(fixed);
  }
  for (const auto& stmt : block->stmts) {
    auto inner = Block::Downcast(stmt);
    if (inner) {
      EvalIndex(inner.get(), tag, value);
      for (auto& idx : inner->idxs) {
        idx.affine = idx.affine.partial_eval(fixed);
      }
    }
  }

  if (it_idx != block->idxs.end()) {
    block->idxs.erase(it_idx);
  }
}

bool UnrollIndexInner(Block* outer, Block* inner, StatementIt it_stmt, const std::string& tag) {
  auto it_idx = GetIndexWithTag(*inner, tag);
  if (it_idx != inner->idxs.end()) {
    for (size_t i = 0; i < it_idx->range; i++) {
      auto clone = CloneBlock(*inner);
      EvalIndex(clone.get(), tag, i);
      for (auto& ref : clone->refs) {
        if (ref.bank_dim && tag == "bank") {  // HACK: can we make this configurable somehow?
          ref.from = str(boost::format("%1%%%%2%") % ref.from % i);
        }
      }
      outer->stmts.insert(it_stmt, clone);
    }
    outer->stmts.erase(it_stmt);
    return true;
  }
  return false;
}

void UnrollIndex(Block* block, const std::string& tag) {
  PreIterate(block, [&](const StatementIt& it) {
    auto inner = Block::Downcast(*it);
    if (inner) {
      if (!UnrollIndexInner(block, inner.get(), it, tag)) {
        UnrollIndex(inner.get(), tag);
      }
    }
  });
}

}  // namespace

void UnrollPass(Block* root, const proto::UnrollPass& options) {
  auto reqs = FromProto(options.reqs());
  PreIterate(root, [&](const StatementIt& it) {
    auto inner = Block::Downcast(*it);
    if (inner) {
      UnrollBlock(root, inner.get(), it, reqs, options.expand_idx());
    }
  });
}

void UnrollIndexPass(Block* root, const proto::UnrollIndexPass& options) {
  auto reqs = FromProto(options.reqs());
  auto idx_reqs = FromProto(options.idx_reqs());
  RunOnBlocks(root, reqs, [&idx_reqs](const AliasMap& map, Block* block) {
    for (const auto& tag : idx_reqs) {
      UnrollIndex(block, tag);
    }
  });
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
