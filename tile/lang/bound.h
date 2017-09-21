#pragma once

#include "OsiClp/OsiClpSolverInterface.hpp"
#include <tuple>
#include <vector>

#include "tile/lang/ops.h"
#include "tile/lang/shape.h"

namespace vertexai {
namespace tile {
namespace lang {

// A range [min, max], ie min <= x <= max
struct Bound {
  int64_t min;  // Smallest value inclusive
  int64_t max;  // Largest value inclusive
};

inline MAKE_LOGGABLE(Bound, b, os) {
  os << "Bound[" << b.min << ", " << b.max << "]";
  return os;
}

// A range for each index
typedef std::map<std::string, Bound> IndexBounds;

// Adds constraints to the contraction forcing every variable used to be an int
Contraction ConstrainIndexVarsToInts(const Contraction& c);

// Gathers boths explicit and implied constraints, and removes dups.
std::vector<RangeConstraint> GatherConstraints(const Contraction& c, const std::vector<TensorShape>& shapes);

// Searches for any parallel constraints and merges them
void MergeParallelConstraints(std::vector<RangeConstraint>* constraints);

// Given two parallel constraints, returns a constraint satisfied by exactly
// those vectors which satisfy both given constraints
RangeConstraint IntersectParallelConstraintPair(const RangeConstraint& constraint1, const RangeConstraint& constraint2);

// Computes the bounds implied by the contraints, and also rewrites remaining contraints
// to be minimal presuming the new set of bounds.  Throws on failure (ie Unbounded)
std::tuple<IndexBounds, std::vector<SimpleConstraint>> ComputeBounds(const std::vector<RangeConstraint>& constraints);

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
