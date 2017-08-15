
#include "tile/lang/loop.h"

#include <boost/format.hpp>

#include <algorithm>

#include "tile/lang/mutil.h"
#include "tile/lang/sembuilder.h"

namespace vertexai {
namespace tile {
namespace lang {

using boost::format;

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
void CodeInfo::thread(uint64_t threads) {
  for (auto& idx : indexes) {
    idx.thread = std::min(threads, NearestPo2(idx.tile));
    threads /= idx.thread;
  }
}

sem::StmtPtr CodeInfo::generate(uint64_t threads, uint64_t div, bool skip_edge, bool order) {
  using namespace sem::builder;  // NOLINT
  // Start with the inner-most block
  sem::StmtPtr cur = inner;
  // Check for thread underrun and correct
  size_t tot_threads = div;
  for (const auto& idx : indexes) {
    tot_threads *= idx.thread;
  }
  if (tot_threads < threads) {
    cur = _If(_("tid") < tot_threads, cur);
  }
  auto r = _Block({});
  // Convert a single thread id to per index thread id, innermost first
  // Also, do per-thread preincrement
  auto tid = _("tid");
  for (int i = 0; i < indexes.size(); i++) {
    // if (indexes[i].thread == 1) { continue; }
    r->append(_Declare({sem::Type::INDEX}, indexes[i].name + "_tid", (tid / div) % indexes[i].thread));
    for (size_t j = 0; j < refs.size(); j++) {
      if (refs[j].strides[i] == 0) {
        continue;
      }
      auto t = _(indexes[i].name + "_tid");
      auto v = _(refs[j].name);
      r->append(v = v + t * refs[j].strides[i]);
    }
    div *= indexes[i].thread;
  }
  // Sort least inefficient loops to the inside
  if (order) {
    std::sort(indexes.begin(), indexes.end(),
              [](const IndexInfo& a, const IndexInfo& b) -> bool { return a.score() > b.score(); });
  }
  // Sort least inefficient loops to the inside
  std::sort(indexes.begin(), indexes.end(),
            [](const IndexInfo& a, const IndexInfo& b) -> bool { return a.score() > b.score(); });
  // Wrap with each outer
  for (int i = indexes.size() - 1; i >= 0; i--) {
    auto b = _Block({});  // Interior of for loop
    ssize_t thread = indexes[i].thread;
    ssize_t edge = indexes[i].total % indexes[i].tile;     // Determine if we have an edge tile
    ssize_t max_loops = RoundUp(indexes[i].tile, thread);  // Compute max loops
    ssize_t max_gid = (indexes[i].total / indexes[i].tile) * indexes[i].tile;
    // Make some variables
    std::string idx_name = indexes[i].name;
    auto idx_lid = _(idx_name + "_lid");
    auto idx_gid = _(idx_name + "_gid");
    auto idx_tid = _(idx_name + "_tid");
    // Add in any breaks we need
    if (edge != 0 && !skip_edge) {       // If we have an edge
      ssize_t low_edge = edge / thread;  // Lowest index we need to break on
      ssize_t high_edge = low_edge + 1;  // Highest index we need to break on
      ssize_t slop = edge % thread;      // Do we break evenly (same of all threads)
      if (slop != 0) {
        // Case for uneven thread break, add up to two breaks
        b->push_back(_If((idx_lid >= low_edge) & (idx_gid == max_gid) & (idx_tid >= slop),
                         continueClause(i, (max_loops - low_edge))));
        if (high_edge != max_loops) {
          b->push_back(_If((idx_lid >= high_edge) & (idx_gid == max_gid), continueClause(i, (max_loops - high_edge))));
        }
      } else {
        // Even thread break
        b->push_back(_If((idx_lid >= low_edge) & (idx_gid == max_gid), continueClause(i, (max_loops - low_edge))));
      }
    }
    if (indexes[i].tile % thread != 0) {
      // This only happens if tile == total, thread only break
      b->push_back(_If((idx_lid >= (max_loops - 1)) & (idx_tid >= (indexes[i].tile % thread)), continueClause(i)));
    }
    // Now add inner code
    b->append(_Declare({sem::Type::INDEX}, idx_name, thread * idx_lid + idx_tid));
    b->append(cur);
    // Step forward
    b->append(increments(i));
    // Build outer block
    auto bo = _Block({});
    bo->append(_For(idx_name + "_lid", max_loops, 1, b));
    // Unwind the changes
    bo->append(increments(i, -max_loops));
    cur = bo;
  }
  r->append(cur);
  return r;
}

std::shared_ptr<sem::Block> CodeInfo::continueClause(int i, ssize_t mul) const {
  using namespace sem::builder;  // NOLINT
  auto r = increments(i, mul);
  r->append(_Continue());
  return r;
}

std::shared_ptr<sem::Block> CodeInfo::increments(int i, ssize_t mul) const {
  using namespace sem::builder;  // NOLINT
  auto r = _Block({});
  if (mul == 0) {
    return r;
  }
  for (size_t j = 0; j < refs.size(); j++) {
    ssize_t hop = mul * indexes[i].thread * refs[j].strides[i];
    auto v = _(refs[j].name);
    if (hop != 0) {
      r->append(v = v + hop);
    }
  }
  return r;
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
