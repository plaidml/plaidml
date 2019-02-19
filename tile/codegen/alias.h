// Copyright 2018, Intel Corporation

#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/util/lookup.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

enum class AliasType {
  None,     // Buffers access unrelated spaces
  Partial,  // Buffers overlap
  Exact,    // Buffers are indentical for every index state
};

struct Extent {
  int64_t min;
  int64_t max;
};

struct AliasInfo {
  stripe::Block* base_block = nullptr;
  stripe::Refinement* base_ref = nullptr;
  std::string base_name;
  std::vector<stripe::Affine> access;
  std::vector<Extent> extents;
  stripe::Location location;
  TensorShape shape;

  static AliasType Compare(const AliasInfo& a, const AliasInfo& b);
  bool IsBanked() const;
};

struct IndexInfo {
  uint64_t range;
  std::string cur_name;
};

class AliasMap {
 public:
  // Constructs a root level alias info
  AliasMap();
  // Construct info for an inner block
  AliasMap(const AliasMap& outer, stripe::Block* block);
  // Lookup an AliasInfo by name
  const AliasInfo& at(const std::string& name) const { return safe_at(info_, name); }
  // Compute statement use count of each buffer
  std::unordered_map<std::string, size_t> RefUseCounts(const stripe::Block& block) const;
  // Attempt to translate an affine into local indexes
  stripe::Affine translate(const stripe::Affine& in) const;
  // Get access to the sources
  const std::map<std::string, stripe::Affine>& idx_sources() const { return idx_sources_; }

 private:
  size_t depth_;                                // How deep is this AliasInfo
  std::map<std::string, AliasInfo> info_;       // Per buffer data
  std::map<std::string, uint64_t> idx_ranges_;  // For each depth-prefixed index, what is it's range?
  std::map<std::string, stripe::Affine>
      idx_sources_;  // For each current index, how is it built from depth-prefixed indexes
};

bool CheckOverlap(const std::vector<Extent>& a_extents, const std::vector<Extent>& b_extents);

template <typename F>
void RunOnBlocksRecurse(const AliasMap& map, stripe::Block* block, const stripe::Tags& reqs, const F& func) {
  if (block->has_tags(reqs)) {
    func(map, block);
  } else {
    for (auto& stmt : block->stmts) {
      auto inner = stripe::Block::Downcast(stmt);
      if (inner) {
        AliasMap inner_map(map, inner.get());
        RunOnBlocksRecurse(inner_map, inner.get(), reqs, func);
      }
    }
  }
}

template <typename F>
void RunOnBlocks(stripe::Block* root, const stripe::Tags& reqs, const F& func) {
  AliasMap base;
  AliasMap root_map(base, root);
  RunOnBlocksRecurse(root_map, root, reqs, func);
}

std::ostream& operator<<(std::ostream& os, const AliasInfo& ai);
std::ostream& operator<<(std::ostream& os, const Extent& extent);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
