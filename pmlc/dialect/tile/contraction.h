// Copyright 2019, Intel Corporation

#pragma once

#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "tile/math/polynomial.h"

#include "pmlc/dialect/stripe/ops.h"
#include "pmlc/dialect/tile/ops.h"

namespace pmlc {
namespace dialect {
namespace tile {

namespace math = vertexai::tile::math;

// A range [min, max], ie min <= x <= max
struct Bound {
  int64_t min;  // Smallest value inclusive
  int64_t max;  // Largest value inclusive
};

using IndexPoly = math::Polynomial<math::Rational>;
using IndexAccess = std::vector<IndexPoly>;
using IndexBounds = std::map<std::string, Bound>;
using RangeConstraints = std::vector<math::RangeConstraint>;
using SimpleConstraints = std::vector<math::SimpleConstraint>;

struct Constraints {
  RangeConstraints constraints;

  void AddTensor(const IndexAccess& access, stripe::TensorType tensorType);

  // Searches for any parallel constraints and merges them
  void MergeParallelConstraints();

  // Computes the bounds implied by the constraints, and also rewrites remaining
  // constraints to be minimal presuming the new set of bounds.
  // Throws on failure (ie Unbounded)
  std::tuple<IndexBounds, SimpleConstraints> ComputeBounds();

  std::set<std::string> VariablesUsed();
};

// enum class AggregationOp {
//   None,
//   Sum,
//   Max,
//   Min,
//   Prod,
//   Assign,
// };

// enum class CombinationOp {
//   None,
//   Mul,
//   Add,
//   Eq,
//   Cond,
// };

struct Contraction {
  explicit Contraction(ContractionOp op);

  std::map<std::string, mlir::Value*> argMap;
  std::vector<IndexAccess> accesses;
  // std::vector<int64_t> size_map;
  // CombinationOp combo_op;
  // AggregationOp agg_op;
  std::vector<math::RangeConstraint> constraints;
  // bool no_defract = false;
  // std::string use_default;

  std::set<std::string> getIndexVars() const;

  // Gathers boths explicit and implied constraints, and removes dups.
  Constraints GatherConstraints(llvm::ArrayRef<stripe::TensorType> shapes) const;

  // Adds constraints to the contraction forcing every variable used to be an
  // integer
  void ConstrainIndexVarsToInts();

  bool NeedReduce() const;
  void ReduceOutputPolynomials(const Constraints& order);

  // Remove any fractional polynomial multipliers (IE, any non-integers).
  void Defractionalize(const Constraints& order);

  std::tuple<IndexBounds, SimpleConstraints> ComputeBounds(llvm::ArrayRef<stripe::TensorType> shapes);
};

}  // namespace tile
}  // namespace dialect
}  // namespace pmlc
