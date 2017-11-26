#pragma once

#include <memory>
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
  std::string name;   // Name of the index
  uint64_t total;     // Full dimension size
  uint64_t tile;      // Tile size (assert <= total, Po2 or total)
  uint64_t thread;    // Number of threads (asset <= tile, Po2)
  int score() const;  // Low score sorts to inside loops
};

// Information about each tensor in the loop
struct TRef {
  std::string name;
  std::vector<int64_t> strides;
};

struct CodeInfo {
  std::vector<IndexInfo> indexes;  // Index names, order, sizes
  std::vector<TRef> refs;          // Output, input1 [, input2]
  sem::BlockPtr inner;

  void thread(uint64_t numThreads);
  sem::BlockPtr generate(uint64_t numThreads, uint64_t div = 1, bool skip_edge = false, bool order = true);

 private:
  sem::BlockPtr increments(int i, ssize_t mul = 1) const;
};

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
