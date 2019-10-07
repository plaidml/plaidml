// Copyright 2018, Intel Corporation

#include "tile/codegen/const_tensor.h"

#include <map>
#include <memory>
#include <set>
#include <string>

#include "base/util/any_factory_map.h"
#include "tile/codegen/tile.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

bool FullyCoveredOut(Block* block, Block* parent) {
  if (block->constraints.size() > 0) {
    return false;
  }
  // Check if the output is fully covered by idxs
  for (const auto& ref : block->ref_outs()) {
    if (ref->agg_op != "" && ref->agg_op != "assign") {
      return false;
    }
    auto ref_it = parent->ref_by_into(ref->from);
    size_t n_dim = ref->access.size();
    for (size_t dim = 0; dim < n_dim; ++dim) {
      const auto& acc_map = ref->access[dim].getMap();
      if (acc_map.size() == 0) {
        if (ref_it->interior_shape.dims[dim].size != 1) {
          return false;
        }
      } else if (acc_map.size() == 1) {
        if (acc_map.begin()->first == "" || acc_map.begin()->second != 1) {
          return false;
        }
        Index* idx = block->idx_by_name(acc_map.begin()->first);
        if (ref_it->interior_shape.dims[dim].size != idx->range) {
          return false;
        }
      } else {
        return false;
      }
    }
  }
  return true;
}

bool AnalyzeConstTensor(Block* block, Block* parent, std::string* tensor, Constant* value) {
  if (block->has_tag("zero") || !FullyCoveredOut(block, parent)) {
    return false;
  }
  // Check if the stmts are either constant-store pattern or constant-assign-store pattern
  int stage = 0;
  std::set<std::string> scalars;
  for (const auto& stmt : block->stmts) {
    switch (stmt->kind()) {
      case StmtKind::Constant: {
        if (stage == 0) {
          ++stage;
          auto constant = Constant::Downcast(stmt);
          scalars.insert(constant->name);
          *value = *(constant.get());
        } else {
          return false;
        }
        break;
      }
      case StmtKind::Intrinsic: {
        auto intrinsic = Intrinsic::Downcast(stmt);
        if (stage == 1 && intrinsic->name == "assign" && scalars.find(intrinsic->inputs[0]) != scalars.end()) {
          ++stage;
          scalars.insert(intrinsic->outputs[0]);
        } else {
          return false;
        }
        break;
      }
      case StmtKind::Store: {
        auto store = Store::Downcast(stmt);
        if ((stage == 1 || stage == 2) && scalars.find(store->from) != scalars.end()) {
          *tensor = store->into;
          ++stage;
        } else {
          return false;
        }
        break;
      }
      default:
        return false;
    }
  }
  return true;
}

void ConstTensor(const AliasMap& alias_map, Block* block, const proto::ConstTensorPass& options) {
  std::map<std::string, Constant> tensor_value;
  auto stmt_it = block->stmts.begin();
  while (stmt_it != block->stmts.end()) {
    auto inner = Block::Downcast(*stmt_it);
    if (!inner) {
      ++stmt_it;
      continue;
    }
    // Check if there is any load that is actually constant
    std::map<std::string, std::string> ref_map;
    for (const auto ref : inner->refs) {
      ref_map.emplace(ref.into(), ref.from);
    }
    std::set<std::string> to_remove;
    auto inner_stmt_it = inner->stmts.begin();
    while (inner_stmt_it != inner->stmts.end()) {
      auto load = Load::Downcast(*inner_stmt_it);
      if (load) {
        const auto ref_name = ref_map.at(load->from);
        auto it = tensor_value.find(ref_name);
        if (it != tensor_value.end()) {
          to_remove.insert(ref_name);
          const auto& value = it->second;
          std::shared_ptr<Constant> new_stmt = (value.type == ConstType::Integer)
                                                   ? std::make_shared<Constant>(load->into, value.iconst)
                                                   : std::make_shared<Constant>(load->into, value.fconst);
          inner->stmts.insert(inner_stmt_it, new_stmt);
          auto old_it = inner_stmt_it;
          ++inner_stmt_it;
          inner->erase_stmt(old_it);
          continue;
        }
      }
      ++inner_stmt_it;
    }
    // Remove the useless refs
    for (const auto& ref : to_remove) {
      auto ref_it = inner->ref_by_from(ref);
      inner->refs.erase(*ref_it);
    }
    // Check if it is a constant tensor
    std::string tensor;
    Constant value("", static_cast<int64_t>(0));
    if (AnalyzeConstTensor(inner.get(), block, &tensor, &value)) {
      tensor_value.emplace(tensor, value);
      auto old_it = stmt_it;
      ++stmt_it;
      block->erase_stmt(old_it);
      continue;
    }
    ++stmt_it;
  }
}

void ConstTensorPass::Apply(CompilerState* state) const {
  auto reqs = stripe::FromProto(options_.reqs());
  RunOnBlocks(
      state->entry(), reqs,
      [this](const AliasMap& alias_map, stripe::Block* block) {  //
        ConstTensor(alias_map, block, options_);
      },
      false);
}

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<ConstTensorPass, proto::ConstTensorPass>::Register();
  return 0;
}();
}  // namespace
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
