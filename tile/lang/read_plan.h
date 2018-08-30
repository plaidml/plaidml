#pragma once

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "tile/lang/flat.h"
#include "tile/lang/semtree.h"

namespace vertexai {
namespace tile {
namespace lang {

// Compute the plan to read per tile data from a source into shared memory
// Also, we merge indexes, for example, if we read from A[i+j], since i + j
// have the same stride, we can treat them identically in certain cases
class ReadPlan {
 private:
  // The original indexes, before merging occurs
  struct OrigIndex {
    OrigIndex(const std::string& _name, int64_t _stride, uint64_t _range)
        : name(_name),  //
          stride(_stride),
          range(_range) {}
    std::string name;         // The name of the index (ie 'i')
    int64_t stride;           // The index's stride in global memory
    uint64_t range;           // The tile-range of the index (ie, size of tile)
    size_t merge_map = 0;     // Which merged index is this index a part of?
    int64_t merge_scale = 0;  // What is the scale of this index in the merged index?
  };
  // The merged indexes
  struct MergedIndex {
    explicit MergedIndex(const OrigIndex& orig)
        : name(orig.name),  //
          stride(std::abs(orig.stride)),
          range(orig.range) {}
    std::string name;           // The name of the merged index (ie, 'i_j')
    int64_t stride;             // The stride of the merged index in global memory
    uint64_t range;             // The tile-range of the merged indedx
    uint64_t zero = 0;          // The position of the 'zero' offset in the merged range
    uint64_t local_stride = 0;  // The stride of this index in local memory
  };

 public:
  // Analyse IO for a set of indexes, strides, and tile-size, as well as cache/phy memory width (in elements)
  ReadPlan(const std::vector<std::string>& names,  //
           const std::vector<int64_t>& strides,    //
           const std::vector<uint64_t>& ranges,    //
           uint64_t mem_width);
  // Return the number of cache lines hit
  uint64_t numLoads() const;
  // Return number of logical elements for this tile size, considering merged indexes
  uint64_t localSize() const;
  // Compute the local load expression
  sem::ExprPtr sharedOffset() const;
  // Compute the global load expression
  sem::ExprPtr globalOffset() const;
  // New version of transfer code generation
  sem::StmtPtr generate(const std::string& to,    //
                        const std::string& from,  //
                        uint64_t threads,         //
                        uint64_t limit,           //
                        uint64_t offset) const;

 private:
  uint64_t mem_width_;               // The minimum read size (due to cache, etc)
  uint64_t local_size_;              // Total amount of local memory required
  uint64_t local_zero_;              // Zero point for all merged indexes
  uint64_t global_zero_;             // Zero point for all merged indexes
  std::vector<OrigIndex> orig_;      // Original indexes
  std::vector<MergedIndex> merged_;  // Merged indexes
  std::vector<size_t> order_;        // The order to transfer merged indexes
};

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
