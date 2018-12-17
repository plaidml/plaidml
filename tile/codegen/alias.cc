// Copyright 2018, Intel Corporation

#include "tile/codegen/alias.h"

#include <boost/format.hpp>

#include "base/util/stream_container.h"
#include "base/util/throw.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

namespace {

Affine UniqifyAffine(const Affine& orig, const std::string& prefix) {
  Affine ret;
  for (const auto& kvp : orig.getMap()) {
    if (kvp.first.empty()) {
      ret += kvp.second;
    } else {
      ret += Affine(prefix + kvp.first, kvp.second);
    }
  }
  return ret;
}

}  // namespace

std::ostream& operator<<(std::ostream& os, const Extent& extent) {
  os << "(" << extent.min << ", " << extent.max << ")";
  return os;
}

bool CheckOverlap(const std::vector<Extent>& ae, const std::vector<Extent>& be) {
  IVLOG(3, boost::format("CheckOverlap: a: '%1%', b: '%2%'") % StreamContainer(ae) % StreamContainer(be));
  if (ae.size() != be.size()) {
    throw std::runtime_error("Incompatible extents");
  }
  bool ret = true;
  for (size_t i = 0; i < ae.size(); i++) {
    ret &= be[i].min <= ae[i].max;
    ret &= ae[i].min <= be[i].max;
  }
  return ret;
}

AliasType AliasInfo::Compare(const AliasInfo& ai, const AliasInfo& bi) {
  IVLOG(3, "Compare: " << ai.base_name << ", " << bi.base_name);
  if (ai.base_name != bi.base_name) {
    return AliasType::None;
  }
  if (ai.shape == bi.shape) {
    if (ai.location.unit.isConstant() && bi.location.unit.isConstant() && ai.location != bi.location) {
      IVLOG(3, boost::format("Different banks, a: %1%, b: %2%") % ai.location % bi.location);
      return AliasType::None;
    }
    if (ai.access == bi.access) {
      IVLOG(3, boost::format("Exact access, a: %1%, b: %2%") % StreamContainer(ai.access) % StreamContainer(bi.access));
      return AliasType::Exact;
    }
    if (!CheckOverlap(ai.extents, bi.extents)) {
      IVLOG(3, "CheckOverlap: None");
      return AliasType::None;
    }
  }
  // TODO: We could compute the convex box enclosing each refinement and then check each
  // dimension to see if there is a splitting plane, and if so, safely declare alias None,
  // but it's unclear that that will happen enough for us to care, so just always return
  // Partial which is conservative.
  return AliasType::Partial;
}

AliasMap::AliasMap() : depth_(0) {}

AliasMap::AliasMap(const AliasMap& outer, stripe::Block* block) : depth_(outer.depth_ + 1) {
  // Make a prefix
  std::string prefix = str(boost::format("d%1%:") % depth_);
  // Make all inner alias data
  for (auto& ref : block->refs) {
    // Setup the place we are going to write to
    AliasInfo& info = info_[ref.into];
    // Check if it's a refinement or a new buffer
    if (ref.dir != RefDir::None) {
      // Get the state from the outer context, fail if not found
      auto it = outer.info_.find(ref.from);
      if (it == outer.info_.end()) {
        throw_with_trace(std::runtime_error(
            str(boost::format("AliasMap::AliasMap: invalid ref.from during aliasing computation: '%1%' (ref: '%2%')") %
                ref.from % ref)));
      }
      // Copy data across
      info.base_block = it->second.base_block;
      info.base_ref = it->second.base_ref;
      info.base_name = it->second.base_name;
      info.access = it->second.access;
    } else {
      // New alloc, initialize from scratch
      info.base_block = block;
      info.base_ref = &ref;
      info.base_name = prefix + ref.into;
      info.access.resize(ref.access.size());
    }
    if (info.access.size() != ref.access.size()) {
      throw_with_trace(std::runtime_error(
          str(boost::format("AliasMap::AliasMap: Mismatched access dimensions on refinement: %1% %2%") %
              info.base_name % ref.into)));
    }
    // Add in indexes from this block
    std::map<std::string, int64_t> min_idxs;
    std::map<std::string, int64_t> max_idxs;
    for (const auto& idx : block->idxs) {
      if (idx.affine.constant()) {
        min_idxs[idx.name] = idx.affine.constant();
        max_idxs[idx.name] = idx.affine.constant();
      } else {
        min_idxs[idx.name] = 0;
        max_idxs[idx.name] = idx.range - 1;
      }
    }
    info.location = ref.location;
    info.extents.resize(ref.access.size());
    for (size_t i = 0; i < ref.access.size(); i++) {
      info.access[i] += UniqifyAffine(ref.access[i], prefix);
      info.extents[i] = Extent{ref.access[i].eval(min_idxs), ref.access[i].eval(max_idxs)};
    }
    IVLOG(5, boost::format("Extents for '%1%' in '%2%': %3%") % ref.into % block->name % StreamContainer(info.extents));

    // Set shape
    info.shape = ref.shape;
  }
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
