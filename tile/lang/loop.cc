
#include "tile/lang/loop.h"

#include "tile/lang/sembuilder.h"
#include "tile/math/util.h"

namespace vertexai {
namespace tile {
namespace lang {

using math::NearestPo2;
using math::RoundUp;

int IndexInfo::score() const {
  if (total % tile == 0 && tile % thread == 0) {
    return 0;
  }
  if (total % tile == 0) {
    return 1;
  }
  if (tile % thread == 0) {
    return 2;
  }
  return 3;
}

void LoopInfo::thread(uint64_t threads) {
  for (auto& idx : indexes) {
    idx.thread = std::min(threads, NearestPo2(idx.tile));
    threads /= idx.thread;
  }
}

sem::StmtPtr LoopInfo::generate(uint64_t threads, uint64_t div, bool skip_edge, size_t select_threshold) {
  using namespace sem::builder;  // NOLINT
  // Start with the inner-most block
  sem::StmtPtr cur = inner;
  // Check for thread underrun and correct
  auto tid = _("tid");
  size_t tot_threads = div;
  for (const auto& idx : indexes) {
    tot_threads *= idx.thread;
  }
  if (tot_threads < threads) {
    cur = _If(tid < tot_threads, cur);
  }
  auto r = _Block({});
  // Convert a single thread id to per index thread id, innermost first
  // Also, do per-thread preincrement
  for (int i = 0; i < indexes.size(); i++) {
    r->append(_Declare({sem::Type::INDEX}, indexes[i].name + "_tid", (tid / div) % indexes[i].thread));
    div *= indexes[i].thread;
  }
  // Sort least inefficient loops to the inside
  std::sort(indexes.begin(), indexes.end(),
            [](const IndexInfo& a, const IndexInfo& b) -> bool { return a.score() > b.score(); });

  // First compute range checks so that they can be composed as we descend to inner loops
  std::vector<sem::ExprPtr> idx_conds;
  for (auto& index : indexes) {
    ssize_t thread = index.thread;
    ssize_t edge = index.total % index.tile;          // Determine if we have an edge tile
    ssize_t max_loops = RoundUp(index.tile, thread);  // Compute max loops
    ssize_t max_gid = (index.total / index.tile) * index.tile;

    std::string idx_name = index.name;
    auto idx_lid = _(idx_name + "_lid");
    auto idx_gid = _(idx_name + "_gid");
    auto idx_tid = _(idx_name + "_tid");
    auto idx_cond = _(idx_name + "_cond");

    if (index.tile % thread != 0) {
      // This only happens if tile == total, thread only break
      index.checks.emplace_back(_LogicalOr(idx_lid < (max_loops - 1), idx_tid < (index.tile % thread)));
    }

    if (edge != 0 && !skip_edge) {       // If we have an edge
      ssize_t low_edge = edge / thread;  // Lowest index we need to break on
      ssize_t high_edge = low_edge + 1;  // Highest index we need to break on
      ssize_t slop = edge % thread;      // Do we break evenly (same of all threads)
      if (slop != 0) {
        // Case for uneven thread break, add up to two breaks
        index.checks.emplace_back(_LogicalOr(idx_lid < low_edge, _LogicalOr(idx_gid != max_gid, idx_tid < slop)));
        if (high_edge != max_loops) {
          index.checks.emplace_back(_LogicalOr(idx_lid < high_edge, idx_gid != max_gid));
        }
      } else {
        // Even thread break
        index.checks.emplace_back(_LogicalOr(idx_lid < low_edge, idx_gid != max_gid));
      }
    }

    if (!index.checks.empty()) {
      idx_conds.push_back(idx_cond);
    }
    index.idx_conds = idx_conds;
  }

  // Wrap with each outer
  size_t element_count = 1;
  for (int i = indexes.size() - 1; i >= 0; i--) {
    ssize_t thread = indexes[i].thread;
    ssize_t max_loops = RoundUp(indexes[i].tile, thread);  // Compute max loops
    element_count *= max_loops;

    // Make some variables
    std::string idx_name = indexes[i].name;
    auto idx_lid = _(idx_name + "_lid");
    auto idx_tid = _(idx_name + "_tid");

    // Potentially wrap the inner code with range checks
    std::shared_ptr<sem::Block> next;
    if (indexes[i].checks.empty()) {
      // Now add inner code
      auto block = _Block({});  // Interior of for loop
      block->append(_Declare({sem::Type::INDEX}, idx_name, thread * idx_lid + idx_tid));
      block->append(cur);
      next = block;
    } else {
      sem::ExprPtr check;
      for (const auto& cond : indexes[i].checks) {
        check = _MaybeLogicalAnd(check, cond);
      }

      next = _Block({});
      auto idx_cond = _Declare(next, {sem::Type::INDEX}, idx_name + "_cond", check);

      // Now add inner code
      auto block = _Block({});  // Interior of for loop
      auto idx_init = thread * idx_lid + idx_tid;
      if (element_count <= select_threshold) {
        sem::ExprPtr acc_cond;
        for (const auto& cond : indexes[i].idx_conds) {
          acc_cond = _MaybeLogicalAnd(acc_cond, cond);
        }
        // Note: Use .get() here to ensure we don't accidentally get a sembuilder operator overload.
        if (!inner_cond.get() && acc_cond.get()) {
          inner_cond = acc_cond;
        }
        auto select = _MaybeSelect(acc_cond, idx_init, _Const(0), {sem::Type::VALUE, DataType::INT32, 1, 0});
        block->append(_Declare({sem::Type::INDEX}, idx_name, select));
        block->append(cur);
        next->append(block);
      } else {
        block->append(_Declare({sem::Type::INDEX}, idx_name, idx_init));
        block->append(cur);
        next->append(_If(idx_cond, block));
      }
    }

    // Build outer block
    auto bo = _Block({});
    if (max_loops == 1) {
      bo->append(_DeclareConst({sem::Type::INDEX}, idx_name + "_lid", 0));
      bo->append(next);
    } else {
      bo->append(_For(idx_name + "_lid", max_loops, 1, next));
    }
    cur = bo;
  }
  r->append(cur);
  return r;
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
