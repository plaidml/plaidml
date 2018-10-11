// Copyright 2018, Intel Corp.

#include "tile/codegen/alias.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

static Affine UniqifyAffine(const Affine& orig, const std::string& prefix) {
  Affine r;
  for (const auto& kvp : orig.getMap()) {
    if (kvp.first.empty()) {
      r += kvp.second;
    } else {
      r += Affine(prefix + kvp.first, kvp.second);
    }
  }
  return r;
}

AliasType AliasInfo::Compare(const AliasInfo& ai, const AliasInfo& bi) {
  if (ai.base_name != bi.base_name) {
    return AliasType::None;
  }
  if (ai.access == bi.access && ai.shape == bi.shape) {
    return AliasType::Exact;
  }
  // TODO: We could compute the convex box enclosing each refinement and then check each
  // dimension to see if there is a splitting plane, and if so, safely declare alias None,
  // but it's unclear that that will happen enough for us to care, so just always return
  // Partial which is conservative.
  return AliasType::Partial;
}

AliasMap::AliasMap() : depth_(0) {}

AliasMap::AliasMap(const AliasMap& outer, const stripe::Block& block) : depth_(outer.depth_ + 1) {
  // Make a prefix
  std::string prefix = std::string("d") + std::to_string(depth_) + ":";
  // Make all inner alias data
  for (const auto& ref : block.refs) {
    // Setup the place we are going to write to
    AliasInfo& info = info_[ref.into];
    // Check if it's a refinement or a new buffer
    if (ref.dir != RefDir::None) {
      // Get the state from the outer context, fail if not found
      auto it = outer.info_.find(ref.from);
      if (it == outer.info_.end()) {
        throw std::runtime_error("AliasMap::AliasMap: invalid ref.from during aliasing computation");
      }
      // Copy data across
      info.base_name = it->second.base_name;
      info.access = it->second.access;
    } else {
      // New alloc, initialize from scratch
      info.base_name = prefix + ref.into;
      info.access.resize(ref.access.size());
    }
    if (info.access.size() != ref.access.size()) {
      throw std::runtime_error("AliasMap::AliasMap: Mismatched sizes on refinement");
    }
    // Add in indexes from this block
    for (size_t i = 0; i < ref.access.size(); i++) {
      info.access[i] += UniqifyAffine(ref.access[i], prefix);
    }
    // Set shape
    info.shape = ref.shape;
  }
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
