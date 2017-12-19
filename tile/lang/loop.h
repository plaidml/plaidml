#pragma once

#include <string>
#include <vector>

#include "base/util/compat.h"
#include "tile/lang/polynomial.h"
#include "tile/lang/semtree.h"

namespace vertexai {
namespace tile {
namespace lang {

// Information about each index in loop
struct IndexInfo {
  std::string name;  // Name of the index
  uint64_t total;    // Full dimension size
  uint64_t tile;     // Tile size (assert <= total, Po2 or total)
  uint64_t thread;   // Number of threads (asset <= tile, Po2)
  std::vector<sem::ExprPtr> checks;
  std::vector<sem::ExprPtr> idx_conds;
  int score() const;  // Low score sorts to inside loops
};

struct LoopInfo {
  std::vector<IndexInfo> indexes;  // Index names, order, sizes
  sem::StmtPtr inner;
  sem::ExprPtr inner_cond;

  void thread(uint64_t numThreads);
  sem::StmtPtr generate(uint64_t numThreads, uint64_t div, bool skip_edge, size_t select_threshold);
};

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
