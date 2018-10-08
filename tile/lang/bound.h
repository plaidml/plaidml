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
std::vector<math::RangeConstraint> GatherConstraints(const Contraction& c, const std::vector<TensorShape>& shapes);

// Searches for any parallel constraints and merges them
void MergeParallelConstraints(std::vector<math::RangeConstraint>* constraints);

// Given two parallel constraints, returns a constraint satisfied by exactly
// those vectors which satisfy both given constraints
math::RangeConstraint IntersectParallelConstraintPair(const math::RangeConstraint& constraint1,
                                                      const math::RangeConstraint& constraint2);

// Computes the bounds implied by the contraints, and also rewrites remaining contraints
// to be minimal presuming the new set of bounds.  Throws on failure (ie Unbounded)
std::tuple<IndexBounds, std::vector<math::SimpleConstraint>> ComputeBounds(
    const std::vector<math::RangeConstraint>& constraints);

// Given polys n1*q + c1 and n2*q + c2 with gcd(n1, n2) = 1, compute d such that
// q + d is integral if and only if n1*q + c1 and n2*q + c2 are both integral
// (or throw if the polys are never both integral).
math::Rational UnifiedOffset(const math::Rational& c1, const math::Rational& c2, const math::Integer& n1,
                             const math::Integer& n2);

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
