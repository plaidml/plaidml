#include "tile/lang/out_plan.h"

#include <assert.h>
#include <boost/range/adaptor/reversed.hpp>

#include <cinttypes>
#include <utility>

#include "base/util/logging.h"
#include "tile/lang/mutil.h"
#include "tile/lang/sembuilder.h"

using std::map;
using std::string;
using std::vector;

namespace vertexai {
namespace tile {
namespace lang {

OutPlan::OutPlan(const FlatContraction& op, const std::vector<uint64_t>& tile, uint64_t threads, uint64_t mem_elems)
    : op_(op), threads_(threads), local_size_(1), outputs_(1), group_dims_({{1, 1, 1}}) {
  using namespace sem::builder;  // NOLINT
  if (mem_elems < 1) {
    mem_elems = 1;
  }
  // Get size into sz
  size_t sz = op.names.size();
  // Copy data into actual structures
  for (size_t i = 0; i < sz; i++) {
    const std::vector<int64_t>& out_stride = op.access[0].strides;
    if (out_stride[i] != 0) {  // Only keep 'output' indexes
      indexes_.emplace_back(op.names[i], i, op.ranges[i], tile[i], out_stride[i]);
    }
  }
  // Sort by stride, then by reverse index.
  std::sort(indexes_.begin(), indexes_.end(), [](const IdxInfo& a, const IdxInfo& b) {
    auto as = std::abs(a.stride);
    auto bs = std::abs(b.stride);
    if (as != bs) {
      return as < bs;
    }
    return b.stride_idx < a.stride_idx;
  });

  // Assign threads
  if (indexes_.size() > 0) {
    // First, assign up to mem_elems of # of threads to low stride entry
    uint64_t low_threads = NearestPo2(std::min(std::min(mem_elems, threads), indexes_[0].tile));
    indexes_[0].threads = low_threads;
    // IVLOG(3, "Setting low threads on index " << indexes_[0].name << " to " << low_threads);
    threads /= low_threads;
    // Now use up the rest of the threads (this is a mediocre huristic)
    while (threads > 1) {
      // Prefer something that divides by two, and then the least threaded
      bool nice_divide = false;
      int min_thread = threads_;
      int pick = -1;
      for (int i = 0; i < indexes_.size(); i++) {
        const auto& idx = indexes_[i];
        if (idx.threads >= idx.tile) continue;
        bool cur_nice_divide = (idx.tile % (idx.threads * 2) == 0);
        int cur_min_thread = idx.threads;
        if ((cur_nice_divide && !nice_divide) || cur_min_thread < min_thread) {
          nice_divide = cur_nice_divide;
          min_thread = cur_min_thread;
          pick = i;
        }
      }
      if (pick == -1) break;
      indexes_[pick].threads *= 2;
      threads /= 2;
      // IVLOG(3, "Multiplying threads on index " << indexes_[pick].name << " to " << indexes_[pick].threads);
    }
  }

  // Compute number of registers used + memory accessed
  local_size_ = threads_;
  for (const auto& idx : indexes_) {
    uint64_t writes = idx.tile;
    if (idx.stride == 1) {
      writes = RoundUp(writes, mem_elems);
    }
    outputs_ *= writes;
    uint64_t parts = RoundUp(idx.tile, idx.threads);
    local_size_ *= parts;
  }

  // Now we assign group ids
  // First, put index number + size into an array
  std::vector<std::pair<size_t, size_t>> to_place;
  for (size_t i = 0; i < indexes_.size(); i++) {
    if (indexes_[i].qout == 1) {
      indexes_[i].base_expr = _Const(0);
    } else {
      to_place.emplace_back(i, indexes_[i].qout);
    }
  }
  // Then sort by size (biggest to smallest) and non-po2 first
  std::sort(to_place.begin(), to_place.end(),
            [](const std::pair<size_t, size_t>& a, const std::pair<size_t, size_t>& b) -> bool {
              return (std::make_tuple(IsPo2(a.second), a.second) < std::make_tuple(IsPo2(b.second), b.second));
            });
  // Now, place into buckets, always use the smallest
  std::array<std::vector<size_t>, 3> buckets;
  for (const auto& p : to_place) {
    size_t which = std::min_element(group_dims_.begin(), group_dims_.end()) - group_dims_.begin();
    buckets[which].push_back(p.first);
    group_dims_[which] *= p.second;
  }
  // Now generate the expressions, and put into place
  for (size_t i = 0; i < 3; i++) {
    size_t cur_below = 1;
    for (const size_t idx_id : boost::adaptors::reverse(buckets[i])) {
      IdxInfo& idx = indexes_[idx_id];
      size_t cur = cur_below * idx.qout;
      sem::ExprPtr expr = _Index(sem::IndexExpr::GROUP, i);
      if (cur != group_dims_[i]) {
        expr = expr % cur;
      }
      if (cur_below != 1) {
        expr = expr / cur_below;
      }
      if (idx.tile > 1) {
        expr = expr * idx.tile;
      }
      idx.base_expr = expr;
      cur_below = cur;
    }
  }
}

std::shared_ptr<sem::Block> OutPlan::initOutput(sem::Type type, sem::ExprPtr value) const {
  using namespace sem::builder;  // NOLINT
  type.array = local_size_ / threads_;
  return _Block({_Declare(type, "agg", value)});
}

std::shared_ptr<sem::Block> OutPlan::initBases() const {
  using namespace sem::builder;  // NOLINT
  auto out = _Block({});
  // For each index, set a _gid variable
  for (const auto& idx : indexes_) {
    out->append(_Declare({sem::Type::INDEX}, idx.name + "_gid", idx.base_expr));
  }
  return out;
}

uint64_t OutPlan::addOutLoops(CodeInfo& ci) const {
  // Add the loops for all output variables
  uint64_t threads = threads_;
  for (const auto& idx : indexes_) {
    ci.indexes.emplace_back(IndexInfo{idx.name, idx.range, idx.tile, idx.threads});
    threads /= idx.threads;
  }
  return threads;
}

sem::ExprPtr OutPlan::regIndex() const {
  using namespace sem::builder;  // NOLINT
  sem::ExprPtr r = _Const(0);
  uint64_t mul = 1;
  for (const auto& idx : indexes_) {
    r = r + _(idx.name + "_lid") * mul;
    mul *= RoundUp(idx.tile, idx.threads);
  }
  return r;
}

uint64_t OutPlan::localSize() const { return local_size_; }

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
