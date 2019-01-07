// Copyright 2018, Intel Corporation

#include "tile/codegen/scalarize.h"

#include <algorithm>

#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

void Scalarize(Block* block, bool recursive) {
  // First, find all potential scalarizable buffers
  std::set<std::string> sbufs;
  for (const auto& ref : block->refs) {
    // Add all locally allocated size one buffers
    if (ref.dir == RefDir::None && ref.interior_shape.elem_size() == 1) {
      sbufs.emplace(ref.into);
    }
  }
  // Remove buffers that are used by inner blocks or primitives
  for (const auto& stmt : block->stmts) {
    if (stmt->kind() == StmtKind::Special || stmt->kind() == StmtKind::Block) {
      for (const auto& name : stmt->buffer_reads()) {
        sbufs.erase(name);
      }
      for (const auto& name : stmt->buffer_writes()) {
        sbufs.erase(name);
      }
    }
  }
  // Remove allocations of sbufs
  auto rem_it = std::remove_if(block->refs.begin(), block->refs.end(),
                               [&](const Refinement& ref) { return sbufs.count(ref.into); });
  block->refs.erase(rem_it, block->refs.end());
  // Now, all we need to do is remove the pointless reads + writes
  // To do so, we track for each 'buffer', which scalar value it currently
  // has, and then for each scalar loaded, what scalar it originally came from
  std::map<std::string, std::string> buf_state;
  std::map<std::string, std::string> scalar_alias;
  auto it = block->stmts.begin();
  while (it != block->stmts.end()) {
    bool keep = true;
    switch ((*it)->kind()) {
      case StmtKind::Store: {
        auto store = Store::Downcast(*it);
        if (sbufs.count(store->into)) {
          // It's a store into a scalarized buffer, record it's effect and erase
          std::string from = store->from;
          if (scalar_alias.count(from)) {
            from = scalar_alias[from];
          }
          buf_state[store->into] = from;
          keep = false;
        }
      } break;
      case StmtKind::Load: {
        auto load = Load::Downcast(*it);
        if (sbufs.count(load->from)) {
          // It's a load from a scalarized buffer, record it's effect and erase
          scalar_alias[load->into] = buf_state[load->from];
          keep = false;
        }
      } break;
      case StmtKind::Intrinsic: {
        auto intr = Intrinsic::Downcast(*it);
        // Update the inputs based on aliases
        for (auto& name : intr->inputs) {
          auto itn = scalar_alias.find(name);
          if (itn != scalar_alias.end()) {
            name = itn->second;
          }
        }
      } break;
      default:
        break;
    }
    if (keep) {
      ++it;
    } else {
      auto to_erase = it;
      ++it;
      block->stmts.erase(to_erase);
    }
  }
  // If recursion was requested, do that
  if (recursive) {
    for (auto& stmt : block->stmts) {
      auto inner = Block::Downcast(stmt);
      if (inner) {
        Scalarize(inner.get(), true);
      }
    }
  }
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
