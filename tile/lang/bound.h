#pragma once

#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "tile/base/shape.h"
#include "tile/bilp/ilp_solver.h"
#include "tile/lang/ops.h"

namespace vertexai {
namespace tile {
namespace lang {

// Adds constraints to the contraction forcing every variable used to be an int
Contraction ConstrainIndexVarsToInts(const Contraction& c);

// Gathers boths explicit and implied constraints, and removes dups.
std::vector<math::RangeConstraint> GatherConstraints(const Contraction& c, const std::vector<TensorShape>& shapes);

// Searches for any parallel constraints and merges them
void MergeParallelConstraints(std::vector<math::RangeConstraint>* constraints);

// Computes the bounds implied by the contraints, and also rewrites remaining contraints
// to be minimal presuming the new set of bounds.  Throws on failure (ie Unbounded)
std::tuple<math::IndexBounds, std::vector<math::SimpleConstraint>> ComputeBounds(
    const std::vector<math::RangeConstraint>& constraints);

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
