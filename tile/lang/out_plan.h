#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "tile/lang/flat.h"
#include "tile/lang/loop.h"
#include "tile/lang/semtree.h"

namespace vertexai {
namespace tile {
namespace lang {

// Compute the plan to clear, (optionally load), and store outputs
// Also handles generation of tile base offsets, workgroup assignment, etc
class OutPlan {
 private:
  struct IdxInfo {
    IdxInfo(const std::string& _name, uint64_t _range, uint64_t _tile, uint64_t _stride)
        : name(_name), range(_range), tile(_tile), stride(_stride), qout((range + tile - 1) / tile), threads(1) {}
    std::string name;        // The name of the index
    uint64_t range;          // It's full range
    uint64_t tile;           // It's tile range
    int64_t stride;          // It's output stride in global memory
    uint64_t qout;           // The quotient of range / tile, ie, number of tiles
    uint64_t threads;        // How many threads to assign to this index
    sem::ExprPtr base_expr;  // The expression to compute base offset from group ids
  };

 public:
  // Construct a new output plan for a given tiling
  OutPlan(const FlatContraction& op, const std::vector<uint64_t>& tile, uint64_t threads, uint64_t mem_elems);
  // Generate code to initialize output locals
  std::shared_ptr<sem::Block> initOutput(sem::Type type, sem::ExprPtr value) const;
  // Generate code to set base offsets from global id
  std::shared_ptr<sem::Block> initBases() const;
  // Generate inner (per tile) loops for each output
  uint64_t addOutLoops(CodeInfo& ci) const;  // NOLINT(runtime/references)
  // Compute the position of the output in the register variable
  sem::ExprPtr regIndex() const;
  // Return number of local element for this tile size
  uint64_t localSize() const;
  // Return number of local element for this tile size
  uint64_t outputs() const { return outputs_; }
  // Return computed group dims
  const std::array<uint64_t, 3>& group_dims() const { return group_dims_; }

 private:
  FlatContraction op_;
  uint64_t threads_;
  uint64_t local_size_;
  uint64_t outputs_;
  std::array<uint64_t, 3> group_dims_;
  std::vector<IdxInfo> indexes_;
};

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
