// Copyright 2019, Intel Corporation

#pragma once

#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "mlir/Pass/Pass.h"

#include "pmlc/dialect/tile/ir/ops.h"
#include "pmlc/util/math/polynomial.h"

namespace pmlc::dialect::tile {

namespace math = util::math;

using IndexPoly = math::Polynomial<math::Rational>;
using IndexAccess = std::vector<IndexPoly>;
using SimpleConstraints = std::vector<math::SimpleConstraint>;
using RangeConstraints = std::vector<math::RangeConstraint>;
using BoundsAndConstraints = std::tuple<math::IndexBounds, SimpleConstraints>;
using Shape = llvm::ArrayRef<int64_t>;

struct Constraints {
  RangeConstraints constraints;

  void AddTensor(const IndexAccess &access, Shape shape);

  // Searches for any parallel constraints and merges them
  void MergeParallelConstraints();

  // Computes the bounds implied by the constraints, and also rewrites remaining
  // constraints to be minimal presuming the new set of bounds.
  // Throws on failure (ie Unbounded)
  BoundsAndConstraints ComputeBounds();

  std::set<std::string> VariablesUsed();
};

struct Contraction {
  explicit Contraction(ContractionOp op);

  BoundsAndConstraints ComputeBounds(llvm::ArrayRef<Shape> shapes);
  void DeduceRangeConstraints();

  std::vector<IndexAccess> accesses;
  SimpleConstraints constraints;
  // During lowering, will transform all constraints to range constraints, which
  // we track in range_constraints
  Constraints range_constraints;

private:
  std::set<std::string> getIndexVars() const;

  // Gathers boths explicit and implied constraints, and removes dups.
  void GatherConstraints(llvm::ArrayRef<Shape> shapes);

  // Adds constraints to the contraction forcing every variable used to be an
  // integer
  void ConstrainIndexVarsToInts();

  bool NeedReduce() const;
  void ReduceOutputPolynomials();

  // Remove any fractional polynomial multipliers (IE, any non-integers).
  void Defractionalize();

private:
  AggregationKind agg_;
};

math::Affine Integerize(const IndexPoly &poly, const math::IndexBounds &bounds);

} // namespace pmlc::dialect::tile
