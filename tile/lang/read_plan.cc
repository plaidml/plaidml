#include "tile/lang/read_plan.h"

#include <assert.h>
#include <boost/algorithm/string/replace.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <map>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

#include "base/util/logging.h"
#include "tile/lang/flat.h"
#include "tile/lang/loop.h"
#include "tile/lang/mutil.h"
#include "tile/lang/parser.h"
#include "tile/lang/sembuilder.h"

using std::map;
using std::string;
using std::vector;

namespace vertexai {
namespace tile {
namespace lang {

ReadPlan::ReadPlan(const std::vector<std::string>& names, const std::vector<int64_t>& strides,
                   const std::vector<uint64_t>& ranges, uint64_t mem_width)
    : mem_width_(mem_width), local_size_(1), local_zero_(0), global_zero_(0) {
  // Sanity check
  size_t size = names.size();
  assert(strides.size() == size);
  assert(names.size() == size);
  assert(ranges.size() == size);

  // Copy in indexes whose strides are not zero
  for (size_t i = 0; i < size; i++) {
    if (strides[i] != 0) {
      orig_.emplace_back(names[i], strides[i], ranges[i]);
    }
  }

  // Sort ranges by absolute global stride
  std::sort(orig_.begin(), orig_.end(),
            [&](const OrigIndex& a, const OrigIndex& b) { return std::abs(a.stride) < std::abs(b.stride); });

  // Merge indexes.  Basically, we copy indexes from orig_ to merged_
  // and merge them as we go.  We merge whenever a new index is an even multiple
  // of another and its ranges overlap.
  for (auto& oi : orig_) {      // For each index
    for (auto& mi : merged_) {  // Look for an index in merged to merge it with
      // Compute the divisor + remainder
      auto div = std::div(std::abs(oi.stride), std::abs(mi.stride));
      // If it's mergeable, adjust the index in question and break
      if (div.rem == 0 && mi.range >= div.quot) {
        mi.range += div.quot * (oi.range - 1);
        mi.name += string("_") + oi.name;
        oi.merge_scale = div.quot * Sign(oi.stride);
        break;
      }
    }
    // Create a new index if this index was not merged.
    if (oi.merge_scale == 0) {
      merged_.emplace_back(oi);
      oi.merge_scale = Sign(oi.stride);
    }
  }

  // Sort merged indexes by odd/even and the largest to smallest
  // We do odd indexes first because they don't need any padding, and when
  // we do pad, we want to minimize it by using larger indexes first
  std::sort(merged_.begin(), merged_.end(), [&](const MergedIndex& a, const MergedIndex b) {
    if (a.range % 2 != b.range % 2) {
      return a.range % 2 == 1;
    }
    return a.range > b.range;
  });

  // Make xfer local strides
  bool did_pad = false;
  for (auto& xi : merged_) {
    // Stride is current size of memory used
    xi.local_stride = local_size_;
    // Make memory use bigger
    local_size_ *= xi.range;
    // If we are even, make things odd again
    if (local_size_ % 2 == 0) {
      local_size_++;
      did_pad = true;
    }
  }
  // Remove the pointless padding on the end
  if (did_pad) {
    local_size_--;
  }

  // Match originals to their associated merged element
  // Also, adjust the 'zero' point for negative strides
  for (auto& oi : orig_) {
    for (size_t i = 0; i < merged_.size(); i++) {
      if (oi.stride == oi.merge_scale * merged_[i].stride) {
        oi.merge_map = i;
        if (oi.merge_scale < 0) {
          merged_[i].zero += std::abs(oi.merge_scale) * (oi.range - 1);
        }
      }
    }
  }

  // Otherwise, compute total zero offset
  for (auto& mi : merged_) {
    local_zero_ += mi.local_stride * mi.zero;
    global_zero_ += mi.stride * mi.zero;
  }

  // Compute the order to thread the merged indexes
  // Start with the 'identity' ordering
  for (size_t i = 0; i < merged_.size(); i++) {
    order_.push_back(i);
  }
  std::sort(order_.begin(), order_.end(), [&](uint64_t a, uint64_t b) {
    const auto& xa = merged_[a];
    const auto& xb = merged_[b];
    // Break things up into 'small-stride' (less than half memory with)
    // and 'large-stride' (everything else)
    bool a_near = (xa.stride < mem_width_ / 2);
    bool b_near = (xb.stride < mem_width_ / 2);
    if (a_near != b_near) {
      // Always prefer small-stride
      return a_near;
    }
    if (a_near) {
      // For small strides, prefer the smaller one
      return xa.stride < xb.stride;
    }
    // Otherwise pick the one that is closer to a power of two
    float ua = static_cast<float>(xa.range) / static_cast<float>(NearestPo2(xa.range));
    float ub = static_cast<float>(xb.range) / static_cast<float>(NearestPo2(xb.range));
    return ua > ub;
  });
}

uint64_t ReadPlan::localSize() const { return local_size_; }

uint64_t ReadPlan::numLoads() const {
  size_t loads = 1;
  if (merged_.size() == 0) {
    return loads;  // Single element
  }
  size_t best_speedup = 1;
  for (const auto& im : merged_) {
    loads *= im.range;
    size_t mem_range = std::min(mem_width_, im.stride * im.range);
    size_t speedup = mem_range / im.stride;
    best_speedup = std::max(best_speedup, speedup);
  }
  loads /= best_speedup;
  return loads;
}

sem::ExprPtr ReadPlan::sharedOffset() const {
  using namespace sem::builder;  // NOLINT
  // Initial offset is zero
  sem::ExprPtr r = _Const(local_zero_);
  for (const auto& oi : orig_) {
    // For each index, determine which merged index it's a part of
    // and multiply by local_stride + scale
    r = r + (merged_[oi.merge_map].local_stride * oi.merge_scale) * _(oi.name);
  }
  return r;
}

sem::ExprPtr ReadPlan::globalOffset() const {
  using namespace sem::builder;  // NOLINT
  // Initial offset is actually zero; we subtract the global_zero_ when we
  // generate the init value for the gid.
  sem::ExprPtr r = _Const(0);
  for (const auto& oi : orig_) {
    // For each index, multiply by its original stride.
    r = r + oi.stride * (_(oi.name) + _(oi.name + "_gid"));
  }
  return r;
}

sem::StmtPtr ReadPlan::generate(const std::string& to, const std::string& from, uint64_t threads, uint64_t limit,
                                uint64_t offset) const {
  using namespace sem::builder;  // NOLINT
  CodeInfo ci;
  sem::ExprPtr lidx = _Const(0);
  sem::ExprPtr gidx = _("gbase");
  for (size_t i = 0; i < merged_.size(); i++) {
    const auto& m = merged_[order_[i]];
    lidx = lidx + int64_t(m.local_stride) * _(m.name);
    gidx = gidx + int64_t(m.stride) * _(m.name);
    ci.indexes.push_back(IndexInfo{m.name, m.range, m.range, 1});
  }
  auto b2 = _Block({});
  b2->append(_Declare({sem::Type::INDEX}, "lidx", lidx));
  b2->append(_Declare({sem::Type::INDEX}, "gidx", gidx));
  sem::StmtPtr assign = (_(to)[_("lidx")] = _(from)[_Clamp(_("gidx"), _Const(-offset), _Const(limit - offset - 1))]);
  b2->append(assign);
  ci.inner = std::move(b2);

  ci.thread(threads);
  auto b = _Block({});
  sem::ExprPtr gbase = _Const(-global_zero_);
  for (const auto& oi : orig_) {
    gbase = gbase + (_(oi.name + "_gid") * oi.stride);
  }
  b->append(_Declare({sem::Type::INDEX}, "gbase", gbase));
  b->merge(ci.generate(threads));
  return _Stmt(std::move(b));
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
