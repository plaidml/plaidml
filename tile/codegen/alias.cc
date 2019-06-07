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

// Determines whether two accesses overlap (assuming base refinement
// and location checks have been performed).  This is used to
// determine the overlap of the internal ranges of two AliasInfos --
// i.e. whether they might overlap for any given instantiation of
// indices.  The implementation currently doesn't check all
// possibilities; it conservatively indicates an overlap in edge
// cases.
bool CheckRelativeOverlap(const AliasInfo& ai, const AliasInfo& bi) {
  IVLOG(4, boost::format("  CheckRelativeOverlap: a: '%1%', b: '%2%'") % ai % bi);
  if (ai.access.size() != bi.access.size()) {
    throw std::runtime_error{"Incompatible accesses"};
  }
  if ((ai.access.size() != ai.shape.dims.size()) || (bi.access.size() != bi.shape.dims.size())) {
    // N.B. This check might as well be an assert; there's no way this
    // should be inconsistent after the AliasInfo's been constructed.
    // But it's good to be careful.
    throw std::runtime_error{"Incompatible access shapes"};
  }
  bool ret = true;
  for (size_t i = 0; ret && i < ai.access.size(); ++i) {
    auto a_affine = ai.access[i];
    auto b_affine = bi.access[i];
    IVLOG(4, "    CheckRelativeOverlap[" << i << "]: Comparing " << a_affine << " with " << b_affine);
    auto& a_map = a_affine.mutateMap();
    auto& b_map = b_affine.mutateMap();
    for (auto a_it = a_map.begin(); a_it != a_map.end();) {
      auto b_it = b_map.find(a_it->first);
      if (b_it == b_map.end()) {
        ++a_it;
        continue;
      }
      auto min_factor = std::min(a_it->second, b_it->second);
      if (b_it->second == min_factor) {
        b_map.erase(b_it);
      } else {
        b_it->second -= min_factor;
      }
      if (a_it->second == min_factor) {
        a_it = a_map.erase(a_it);
      } else {
        a_it->second -= min_factor;
        ++a_it;
      }
    }
    IVLOG(4, "    CheckRelativeOverlap[" << i << "]: Simplified to " << a_affine << " and " << b_affine);
    if (!a_affine.isConstant() || !b_affine.isConstant()) {
      // TODO: Figure out how to compute the correct answer when the
      // resulting affines aren't constant.  In this situation, the
      // refinements are moving relative to each other across
      // different instantiations; we'd need to tell whether this ever
      // ends up with the refinements' colliding.
      return true;
    }
    IVLOG(4, "    CheckRelativeOverlap[" << i << "]: Simplified to " << a_affine << " and " << b_affine);
    std::int64_t a_first = a_affine.constant();
    std::int64_t a_limit = a_first + ai.shape.dims.at(i).size;
    std::int64_t b_first = b_affine.constant();
    std::int64_t b_limit = b_first + bi.shape.dims.at(i).size;
    ret &= b_first < a_limit;
    ret &= a_first < b_limit;
  }
  IVLOG(4, boost::format("  CheckRelativeOverlap: a: '%1%', b: '%2%' => %3%") % ai % bi % ret);
  return ret;
}

}  // namespace

std::ostream& operator<<(std::ostream& os, const AliasInfo& ai) {
  os << "(" << ai.base_name;
  os << ", " << ai.location;
  os << ", " << StreamContainer(ai.access);
  os << ", " << ai.shape;
  os << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Extent& extent) {
  os << "(" << extent.min << ", " << extent.max << ")";
  return os;
}

bool CheckOverlap(const std::vector<Extent>& ae, const std::vector<Extent>& be) {
  IVLOG(4, boost::format("  CheckOverlap: a: '%1%', b: '%2%'") % StreamContainer(ae) % StreamContainer(be));
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
  IVLOG(3, "AliasInfo::Compare> a: " << ai.base_name << ", b: " << bi.base_name);
  IVLOG(4, "  a: " << ai);
  IVLOG(4, "  b: " << bi);
  if (ai.base_name != bi.base_name) {
    IVLOG(3, "  Different base tensors");
    return AliasType::None;
  }

  if (ai.location != bi.location) {
    IVLOG(3, boost::format("  Different banks, a: %1%, b: %2%") % ai.location % bi.location);
    return AliasType::None;
  }
  if (ai.access == bi.access) {
    IVLOG(3, boost::format("  Exact access, a: %1%, b: %2%") % StreamContainer(ai.access) % StreamContainer(bi.access));
    return AliasType::Exact;
  }
  if (!CheckRelativeOverlap(ai, bi)) {
    IVLOG(3, "  No overlap");
    return AliasType::None;
  }
  // TODO: We could compute the convex box enclosing each refinement and then check each
  // dimension to see if there is a splitting plane, and if so, safely declare alias None,
  // but it's unclear that that will happen enough for us to care, so just always return
  // Partial which is conservative.
  IVLOG(3, "  Partial");
  return AliasType::Partial;
}

bool AliasInfo::IsBanked() const { return !!base_ref->bank_dim; }

Affine AliasInfo::flat() const {
  Affine flat;
  for (size_t i = 0; i < shape.dims.size(); i++) {
    flat += access[i] * shape.dims[i].stride;
  }
  return flat;
}

AliasMap::AliasMap() : depth_(0), this_block_(nullptr), parent_block_(nullptr), parent_alias_map_(nullptr) {}

AliasMap::AliasMap(const AliasMap& outer, stripe::Block* block) : depth_(outer.depth_ + 1), parent_alias_map_(&outer) {
  idx_ranges_ = outer.idx_ranges_;
  this_block_ = block;
  parent_block_ = outer.this_block();
  // Make a prefix
  std::string prefix = str(boost::format("d%1%:") % depth_);
  // Get all the index data for new indexes
  for (const auto& idx : block->idxs) {
    if (idx.affine == stripe::Affine()) {
      // If it's a normal index, add it's information
      idx_ranges_[prefix + idx.name] = idx.range;
      idx_sources_[idx.name] = Affine(prefix + idx.name);
    } else {
      idx_sources_[idx.name] = idx.affine.sym_eval(outer.idx_sources_);
    }
  }
  // Make all inner alias data
  for (auto& ref : block->refs) {
    // Setup the place we are going to write to
    AliasInfo& info = info_[ref.into()];
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
      info.location = AddDeviceUnits(it->second.location, ref.location);
    } else {
      // New alloc, initialize from scratch
      info.base_block = block;
      info.base_ref = &ref.mut();
      info.base_name = prefix + ref.into();
      info.access.resize(ref.access.size());
      info.location = ref.location;
    }
    if (info.access.size() != ref.access.size()) {
      throw_with_trace(std::runtime_error(
          str(boost::format("AliasMap::AliasMap: Mismatched access dimensions on refinement: %1% %2%") %
              info.base_name % ref.into())));
    }
    info.extents.resize(ref.access.size());
    for (size_t i = 0; i < ref.access.size(); i++) {
      info.access[i] += UniqifyAffine(ref.access[i], prefix);
      Extent& ext = info.extents[i];
      ext.min = 0;
      ext.max = 0;
      for (const auto& kvp : info.access[i].getMap()) {
        if (kvp.first == "") {
          ext.min += kvp.second;
          ext.max += kvp.second;
        } else {
          if (kvp.second > 0) {
            ext.max += (idx_ranges_[kvp.first] - 1) * kvp.second;
          } else {
            ext.min += (idx_ranges_[kvp.first] - 1) * kvp.second;
          }
        }
      }
      ext.max += ref.interior_shape.dims[i].size - 1;
    }
    IVLOG(5,
          boost::format("Extents for '%1%' in '%2%': %3%") % ref.into() % block->name % StreamContainer(info.extents));
    // Set shape
    info.shape = ref.interior_shape;
  }
}

std::unordered_map<std::string, size_t> AliasMap::RefUseCounts(const Block& block) const {
  // Compute statement use count of each buffer
  std::unordered_map<std::string, size_t> use_count;
  for (const auto& stmt : block.stmts) {
    std::set<std::string> buf_use;
    for (const auto& str : stmt->buffer_reads()) {
      buf_use.emplace(str);
    }
    for (const auto& str : stmt->buffer_writes()) {
      buf_use.emplace(str);
    }
    for (const auto& str : buf_use) {
      use_count[str]++;
    }
  }
  return use_count;
}

stripe::Affine AliasMap::translate(const stripe::Affine& in) const {
  IVLOG(3, "AliasMap::translate in = " << in);
  Affine cur = in;
  for (const auto& kvp : idx_ranges_) {
    if (kvp.second == 1) {
      cur.mutateMap().erase(kvp.first);
    }
  }
  IVLOG(3, "AliasMap::after removing 'irrelevant' indexes = " << cur);
  Affine out;
  for (const auto& kvp : idx_sources_) {
    IVLOG(4, "  " << kvp.first << " = " << kvp.second);
    int64_t mul = 0;
    for (const auto& kvp2 : kvp.second.getMap()) {
      if (mul == 0 && cur.getMap().count(kvp2.first)) {
        mul = cur.getMap().at(kvp2.first) / kvp2.second;
        break;
      }
    }
    out += Affine(kvp.first, mul);
    cur -= kvp.second * mul;
    IVLOG(4, "  mul = " << mul << ", cur = " << cur);
  }
  if (!cur.isConstant()) {
    throw std::runtime_error("AliasMap::translate, unable to translate " + in.toString());
  }
  out += cur.constant();
  IVLOG(3, "  out = " << out);
  return out;
}

void AliasMap::AddConstraintForIndex(stripe::Block* block,         //
                                     const AliasInfo& alias_info,  //
                                     size_t idx,                   //
                                     const std::string& idx_name,  //
                                     bool idx_passthru) const {
  int64_t top_index = alias_info.base_ref->interior_shape.dims[idx].size - 1;
  bool underflow = alias_info.extents[idx].min < 0;
  bool overflow = alias_info.extents[idx].max > top_index;
  if (underflow || overflow) {
    IVLOG(3, "AddConstraintForIndex: " << alias_info.base_name << ", " << idx_name);
    IVLOG(3, "extents = " << alias_info.extents[idx] << ", top_index = " << top_index);
    if (idx_name != "") {
      std::string global_idx_name = block->unique_idx_name(idx_name);
      block->idxs.emplace_back(Index{global_idx_name, 1, translate(alias_info.access[idx])});
      if (underflow) {
        block->constraints.push_back(idx_passthru ? Affine(global_idx_name)
                                                  : (Affine(global_idx_name) + Affine(idx_name)));
      }
      if (overflow) {
        block->constraints.push_back(idx_passthru ? (Affine(top_index) - Affine(global_idx_name))
                                                  : (Affine(top_index) - Affine(global_idx_name) - Affine(idx_name)));
      }
    }
    else {
      Affine exp = translate(alias_info.access[idx]);
      if (underflow) {
        block->constraints.push_back(exp);
      }
      if (overflow) {
        block->constraints.push_back(Affine(top_index) - exp);
      }
    }
  }
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
