// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/contraction.h"

#include <limits>

#include "llvm/Support/FormatVariadic.h"

#include "mlir/IR/Matchers.h"
#include "mlir/Support/DebugStringHelper.h"

#include "base/util/logging.h"
#include "pmlc/util/util.h"
#include "tile/bilp/ilp_solver.h"
#include "tile/math/basis.h"

namespace bilp = vertexai::tile::bilp;

using mlir::AffineExpr;
using mlir::AffineExprKind;
using mlir::ArrayAttr;

using vertexai::tile::math::Integer;
using vertexai::tile::math::RangeConstraint;
using vertexai::tile::math::Rational;
using vertexai::tile::math::SimpleConstraint;

namespace pmlc::dialect::tile {

// TODO: What's with all these static functions?

static bool IsImplied(const SimpleConstraint& constraint, const IndexBounds& bounds) {
  auto worst = constraint.poly.constant();
  for (const auto& [key, value] : constraint.poly.getMap()) {
    if (key.empty()) {
      continue;
    }
    if (value < 0) {
      worst += value * bounds.find(key)->second.min;
    } else {
      worst += value * bounds.find(key)->second.max;
    }
  }
  return (worst <= constraint.rhs);
}

static Rational UnifiedOffset(const Rational& c1, const Rational& c2, const Integer& n1, const Integer& n2) {
  std::set<Rational> offsets;
  if (n1 > std::numeric_limits<std::size_t>::max() || n2 > std::numeric_limits<std::size_t>::max()) {
    throw std::out_of_range("Cannot unify offset when relative quotient exceeds size_t.");
  }
  for (size_t i = 0; i < math::Abs(n1); ++i) {
    offsets.insert(std::end(offsets), math::FracPart((c1 + i) / n1));
  }
  for (size_t j = 0; j < math::Abs(n2); ++j) {
    Rational offset = math::FracPart((c2 + j) / n2);
    if (offsets.count(offset)) {
      return offset;
    }
  }
  IVLOG(1, "Failed to compute UnifiedOffset(" << c1 << ", " << c2 << ", " << n1 << ", " << n2 << ").");
  throw std::runtime_error("Merging constraints with empty intersection.");
}

static RangeConstraint IntersectParallelConstraintPairInner(  //
    const RangeConstraint& constraint1,                       //
    const RangeConstraint& constraint2) {
  // The primary algorithm for combining two parallel constraints into one. See
  // merge-parallel.tex in /tile/lang for more details.
  // Assumes the RangeConstraints have been validated to have positive ranges, _or_ that
  // for constraint2 specifically it represents a one sided constraint by having an
  // infinite range parameter represented as a range value of -1
  Rational ratio = constraint1.poly.tryDivide(constraint2.poly, true);
  if (ratio == 0) {
    throw std::invalid_argument("Parameters of IntersectParallelConstraintPair must be parallel");
  }
  Integer n1 = numerator(ratio);
  Integer n2 = denominator(ratio);
  Rational c1 = constraint1.poly.constant();
  Rational c2 = constraint2.poly.constant();
  // d is the fractional part of the offset of merged constraint polynomial
  Rational d = UnifiedOffset(c1, c2, n1, n2);
  // Range unification requires solving the following equations for q:
  //    n1*q + c1 = 0           n2*q + c2 = 0
  //    n1*q + c1 = r1 - 1      n2*q + c2 = r2 - 1
  Rational q1_low = math::Min(-c1 / n1, (constraint1.range - 1 - c1) / n1);
  Rational q1_hi = math::Max(-c1 / n1, (constraint1.range - 1 - c1) / n1);
  Integer lower_bound;
  Integer upper_bound;
  if (constraint2.range > 0) {
    Rational q2_low = math::Min(-c2 / n2, (constraint2.range - 1 - c2) / n2);
    Rational q2_hi = math::Max(-c2 / n2, (constraint2.range - 1 - c2) / n2);
    lower_bound = math::Max(math::Ceil(q1_low + d), math::Ceil(q2_low + d));
    upper_bound = math::Min(math::Floor(q1_hi + d), math::Floor(q2_hi + d));
  } else if (constraint2.range == -1) {
    // Same calculations of lower/upper_bound as above, but with constraint2.range == infinity
    Rational q2_low = -c2 / n2;
    lower_bound = math::Max(math::Ceil(q1_low + d), math::Ceil(q2_low + d));
    upper_bound = math::Floor(q1_hi + d);
  } else {
    throw std::runtime_error("Given constraint with empty range in IntersectParallelConstraintPair: " +
                             to_string(constraint2));
  }
  Rational merged_offset = -lower_bound + d;
  Integer range = upper_bound - lower_bound + 1;
  if (range <= 0) {
    throw std::runtime_error("Merging constraints with empty intersection: " + to_string(constraint1) + ", " +
                             to_string(constraint2));
  }
  if (range > INT64_MAX) {
    throw std::out_of_range("Bound range in IntersectParallelConstraintPair overflows int64.");
  }
  int64_t r = (int64_t)range;
  IndexPoly p(constraint1.poly / n1);
  p.setConstant(merged_offset);
  return RangeConstraint(p, r);
}

static RangeConstraint IntersectParallelConstraintPair(  //
    const RangeConstraint& constraint1,                  //
    const SimpleConstraint& constraint2) {
  // Combines two parallel constraints into one
  IVLOG(5, "Merging the parallel constraints " << constraint1 << ", " << constraint2);
  if (constraint1.range <= 0) {
    throw std::runtime_error("Given constraint with empty range in IntersectParallelConstraintPair: " +
                             to_string(constraint1));
  }
  return IntersectParallelConstraintPairInner(constraint1, RangeConstraint(constraint2.rhs - constraint2.poly, -1));
}

static RangeConstraint IntersectParallelConstraintPair(  //
    const RangeConstraint& constraint1,                  //
    const RangeConstraint& constraint2) {
  // Combines two parallel constraints into one
  IVLOG(5, "Merging the parallel constraints " << constraint1 << ", " << constraint2);
  if (constraint1.range <= 0) {
    throw std::runtime_error("Given constraint with empty range in IntersectParallelConstraintPair: " +
                             to_string(constraint1));
  }
  if (constraint2.range <= 0) {
    throw std::runtime_error("Given constraint with empty range in IntersectParallelConstraintPair: " +
                             to_string(constraint2));
  }
  return IntersectParallelConstraintPairInner(constraint1, constraint2);
}

// TODO: Cull?
// static RangeConstraint IntersectParallelConstraintPair(  //
//     const SimpleConstraint& constraint1,                 //
//     const RangeConstraint& constraint2) {
//   // Combines two parallel constraints into one
//   return IntersectParallelConstraintPair(constraint2, constraint1);
// }

// TODO: The Intersect...Constraint... functions should probably be moved to tile/math/polynomial.h

// TODO: Need unit tests
static RangeConstraint IntersectOpposedSimpleConstraints(  //
    const SimpleConstraint& constraint1,                   //
    const SimpleConstraint& constraint2) {
  // Combines two SimpleConstraints which are parallel and bounding in opposite directions into
  // a single equivalent RangeConstraint
  Rational ratio = constraint1.poly.tryDivide(constraint2.poly, true);
  if (ratio >= 0) {
    throw std::invalid_argument(
        "Parameters of IntersectOpposedSimpleConstraints must be parallel and in opposite directions");
  }
  // We know constraint1.poly <= constraint1.rhs and also constraint2.poly <= constraint2.rhs
  // Could rewrite as 0 <= -constraint1.poly + constraint1.rhs and also 0 <= -constraint2.poly + constraint2.rhs
  // Rewrite: constraint1.poly = p1 + c1 (where p1 has no constant part), constraint1.rhs = rhs1; similarly for 2
  // So 0 <= -p1 - c1 + rhs1 and also 0 <= -p2 - c2 + rhs2
  // and since a = ratio is negative, and p2 * a = p1, get
  //   0 >= -a * p2 - a * c2 + a * rhs2
  //   0 >= -p1 - a * c2 + a * rhs2
  //   -c1 + rhs1 >= -p1 - a * c2 + a * rhs2 - c1 + rhs1
  //   -c1 + rhs1 + a * c2 - a * rhs2 >= -p1 -c1 + rhs1
  // So, merge into
  //   0 <= -p1 - c1 + rhs1 <= -c1 + rhs1 + a * c2 - a * rhs2
  // (noting that this is just for the bounds, not the integrality)
  // And so use this to make a range constraint from constraint1 and then call to IntersectParallelConstraintPair
  // Note that if the range is too generous the IntersectParallel call will fix it, so don't bother with a - 1
  // This may have some duplication of work, but I think this is too tiny for that to matter for overall perf
  auto merged_poly1 = -constraint1.poly + constraint1.rhs;
  auto merged_const1 = -constraint1.poly.constant() + constraint1.rhs;
  auto merged_const2 = -constraint2.poly.constant() + constraint2.rhs;
  auto range = merged_const1 - ratio * merged_const2;
  if (range > INT64_MAX) {
    throw std::out_of_range("Bound range in IntersectOpposedSimpleConstraints overflows int64.");
  }
  int64_t r = (int64_t)range;
  return IntersectParallelConstraintPair(RangeConstraint(merged_poly1, r), constraint2);
}

void Constraints::AddTensor(const IndexAccess& access, stripe::TensorType tensorType) {
  // TODO: Redo for `constraints` being `SimpleConstraint`s
  auto shape = tensorType.getShape();
  if (access.size() != shape.size()) {
    throw std::runtime_error(llvm::formatv("Indexes != dimensions: {0} != {1}", access.size(), shape.size()).str());
  }
  for (size_t i = 0; i < access.size(); i++) {
    constraints.emplace_back(access[i], shape[i].size);
  }
}

void Constraints::MergeParallelConstraints() {
  auto i = constraints.begin();
  while (i != constraints.end()) {
    // Remove trivially true constraints, fail for trivially false constraints
    // By trivially true/false, I mean constraints of the form 0 <= c < r,
    // where c is a number rather than a polynomial.
    if (i->poly.GetNonzeroIndex().empty()) {
      if (0 <= i->poly.constant() && i->poly.constant() < i->range) {
        // Constraint is trivially true; remove and continue
        if (i == constraints.begin()) {
          constraints.erase(i);
          i = constraints.begin();
          continue;
        } else {  // i-- exists
          i--;
          constraints.erase(i + 1);
          i++;  // This is the same value of i we would have checked next if not deleting
          continue;
        }
      } else {
        // Constraint is trivially false; fail indicating no solutions
        std::string ErrorMessage = "Error: Always false constraint given to MergeParallelConstraints.";
        ErrorMessage += "\nConstraint poly: " + i->poly.toString();
        ErrorMessage += "\nConstraint range: " + std::to_string(i->range);
        throw std::invalid_argument(ErrorMessage);
      }
    }
    for (auto j = i + 1; j != constraints.end(); ++j) {
      if (i->IsParallel(*j)) {
        (*i) = IntersectParallelConstraintPair(*i, *j);

        // Decrement j so it stays valid after erase, then erase where j
        // used to be. Incrementing this j at the end of the loop body
        // gives the same element as if we hadn't deleted the original j
        //
        // Slightly inefficient to repeatedly erase; could instead
        // create a list of iterators pointing at things to erase, then
        // erase them all at the end. But there's some added complexity
        // to that and I don't think it gains us much speed, so I'm
        // not doing that (for now? [TODO perf (minor)])
        j--;  // must exist since i exists
        constraints.erase(j + 1);
      }
    }
    ++i;
  }
}

std::set<std::string> Constraints::VariablesUsed() {
  // Returns all the variables appearing in the constraints
  std::set<std::string> ret;
  for (const auto& constraint : constraints) {
    for (const auto& [key, value] : constraint.poly.getMap()) {
      if (key.size()) {  // Do nothing for constant term
        ret.emplace(key);
      }
    }
  }
  return ret;
}

// TODO(T133): Check size of integer programming problem to prevent slowdown
BoundsAndConstraints Constraints::ComputeBounds() {
  auto vars = VariablesUsed();

  // Run the solver for each variable min + max
  bilp::ILPSolver solver;
  IndexBounds out;
  IndexAccess objectives;
  for (const std::string& var : vars) {
    objectives.emplace_back(var);
    objectives.emplace_back(var, -1);
  }
  std::map<IndexPoly, bilp::ILPResult> result = solver.batch_solve(constraints, objectives);
  for (const auto& [key, value] : result) {
    // ILPResult lists the objective for each requested optimization. Since we
    // used a monomial for each objective, GetNonzeroIndex returns the name of
    // the variable. Then we grab its coefficient to see if we were requesting
    // minimization or maximization
    std::string var = key.GetNonzeroIndex();
    if (key[var] == 1) {
      out[var].min = static_cast<int64_t>(value.obj_val);
    } else if (key[var] == -1) {
      out[var].max = static_cast<int64_t>(-value.obj_val);
    } else {
      throw std::runtime_error("Internal error: unexpected ILP objective type");
    }
  }

  // Remove constraints which are implied
  SimpleConstraints remaining;
  for (const auto& constraint : constraints) {
    auto lower = constraint.lowerBound();
    if (!IsImplied(lower, out)) {
      remaining.push_back(lower);
    }
    auto upper = constraint.upperBound();
    if (!IsImplied(upper, out)) {
      remaining.push_back(upper);
    }
  }

  return std::tie(out, remaining);
}

static IndexPoly MakePoly(ContractionOp op, AffineExpr expr) {
  IVLOG(3, "MakePoly: " << mlir::debugString(expr));
  if (auto dimExpr = expr.dyn_cast<mlir::AffineDimExpr>()) {
    auto idxNames = op.getAttrOfType<ArrayAttr>("idxs");
    if (idxNames) {
      auto attr = idxNames.getValue()[dimExpr.getPosition()];
      auto name = attr.cast<StringAttr>().getValue();
      return IndexPoly{name.str()};
    }
    auto name = llvm::formatv("x{0}", dimExpr.getPosition());
    return IndexPoly{name.str()};
  }
  if (auto constExpr = expr.dyn_cast<mlir::AffineConstantExpr>()) {
    return IndexPoly{constExpr.getValue()};
  }
  if (auto binaryExpr = expr.dyn_cast<mlir::AffineBinaryOpExpr>()) {
    switch (binaryExpr.getKind()) {
      case AffineExprKind::Add:
        return MakePoly(op, binaryExpr.getLHS()) + MakePoly(op, binaryExpr.getRHS());
      case AffineExprKind::Mul: {
        auto lhs = MakePoly(op, binaryExpr.getLHS());
        auto rhs = MakePoly(op, binaryExpr.getRHS());
        if (!rhs.isConstant()) {
          // AffineExpr should gaurantee that constants are on the RHS
          throw std::runtime_error(
              llvm::formatv("Non-linear polynomial: {0} * {1}", lhs.toString(), rhs.toString()).str());
        }
        return lhs * rhs.constant();
      }
      case AffineExprKind::FloorDiv: {
        auto lhs = MakePoly(op, binaryExpr.getLHS());
        auto rhs = MakePoly(op, binaryExpr.getRHS());
        if (!rhs.isConstant()) {
          throw std::runtime_error(
              llvm::formatv("Divisor of polynomials must be a constant: {0} / {1}", lhs.toString(), rhs.toString())
                  .str());
        }
        return MakePoly(op, binaryExpr.getLHS()) / rhs.constant();
      }
      default:
        throw std::runtime_error("Unsupported AffineBinaryOpExpr");
    }
  }
  throw std::runtime_error("Invalid AffineExpr");
}

static IndexAccess ConvertAffineMap(ContractionOp op, mlir::AffineMap map) {
  IndexAccess dims;
  for (auto expr : map.getResults()) {
    dims.emplace_back(MakePoly(op, expr));
  }
  return dims;
}

Contraction::Contraction(ContractionOp op) {
  accesses.emplace_back(ConvertAffineMap(op, op.sink()));
  for (auto src : op.srcs()) {
    auto map = src.cast<AffineMapAttr>().getValue();
    accesses.emplace_back(ConvertAffineMap(op, map));
  }

  // TODO: enable this once we can use SimpleConstraints directly
  // if (op.cons().hasValue()) {
  //   for (auto cons : op.cons().getValue().getConstraints()) {
  //     constraints.emplace_back(MakePoly(op, cons), 0);
  //   }
  // }
}

void Contraction::GatherConstraints(llvm::ArrayRef<stripe::TensorType> shapes) {
  // Sanity check the shapes
  if (shapes.size() != accesses.size()) {
    throw std::runtime_error(
        llvm::formatv("Shape mismatch during constraint gathering: {0} vs {1}", shapes.size(), accesses.size()).str());
  }
  // Add constraints to keep each access in-bounds
  // TODO: Do we actually need this each run of GatherConstraints? Seems like first time should be sufficient...
  for (size_t i = 0; i < accesses.size(); i++) {
    range_constraints.AddTensor(accesses[i], shapes[i]);
  }
  std::stable_sort(range_constraints.constraints.begin(), range_constraints.constraints.end(),
                   [](const auto& x, const auto& y) {  //
                     return (x.range < y.range);
                   });
}

void Contraction::ConstrainIndexVarsToInts() {
  // This implementation makes RangeConstraints, so can't be used except during lowering
  const int32_t kBoundWidth = 1000000000;
  for (const auto& var : getIndexVars()) {
    if (!var.empty()) {  // Constant components not used
      range_constraints.constraints.emplace_back(RangeConstraint(IndexPoly(var) + kBoundWidth / 2, kBoundWidth));
    }
  }
}

std::set<std::string> Contraction::getIndexVars() const {
  // Note that this validates that `constraints` don't have variables that don't appear in accesses,
  // not that `range_constraints` don't have such variables. So if using after the construction of
  // range_constraints, make sure nothing will have been transformed
  std::set<std::string> vars;
  for (const auto& access : accesses) {
    for (const auto& poly : access) {
      for (const auto& [key, value] : poly.getMap()) {
        vars.insert(key);
      }
    }
  }
  // Valdiate that no variables appear only in the constraints
  // Pre-add the 'constant' variable (which is always valid)
  vars.insert("");
  for (const auto& constraint : constraints) {
    for (const auto& [key, value] : constraint.poly.getMap()) {
      // If there is a variable used in a constraint and that variable doesn't
      // appear in the list of variables from the tensors, then it's a variable
      // that appears only in the constraints, and we need to throw.
      if (vars.find(key) == vars.end()) {
        std::ostringstream ss;
        ss << "Contraction::getIndexAndOutputVars: Variable '" << key
           << "' appears only in constraints of contraction:\n"
           << " Tensors:";
        for (const auto& access : accesses) {
          ss << " {";
          for (const auto& poly : access) {
            ss << poly.toString() << ", ";
          }
          ss.seekp(-2, ss.cur);
          ss << "},";
        }
        ss.seekp(-1, ss.cur);
        ss << "\nConstraints:";
        for (const auto& cons_error : constraints) {
          ss << " { Poly: " << cons_error.poly.toString();
          ss << ", RHS: " << std::to_string(cons_error.rhs) << " }";
        }
        throw std::runtime_error(ss.str());
      }
    }
  }
  // Erase constant
  vars.erase("");
  return vars;
}

static IndexPoly ConvertVariables(     //
    const IndexPoly& in,               //
    llvm::ArrayRef<std::string> vars,  //
    llvm::ArrayRef<IndexPoly> polys) {
  IndexPoly out;
  for (size_t i = 0; i < vars.size(); i++) {
    out += in[vars[i]] * polys[i];
  }
  out += in.constant();
  return out;
}

static IndexPoly ConvertPoly(                       //
    IndexPoly in,                                   //
    const std::map<std::string, IndexPoly>& polys,  //
    bool transform_constant = false) {
  IndexPoly out;
  for (const auto& [key, value] : in.getMap()) {
    if (key == "" && !transform_constant) {
      out += value;
    } else {
      auto it = polys.find(key);
      if (it == polys.end()) {
        throw std::runtime_error("Invalid polynomial conversion");
      }
      out += value * it->second;
    }
  }
  return out;
}

void Contraction::ReduceOutputPolynomials() {
  // Note that this must be run between when `range_constraints` is constructed and any further transforms of accesses,
  // due to the use of getIndexVars.
  // First, we find all of our index variables
  auto indexVars = getIndexVars();

  // Now we construct a set of 'rewrite' variables for each linearly independent
  // output IndexPoly
  math::BasisBuilder basis;
  for (const auto& poly : accesses[0]) {
    // Maybe add it to the equation list
    basis.addEquation(poly);
  }

  // Next, fill in from contraints until we have as many as variables or we run
  // out of options
  for (const auto& constraint : range_constraints.constraints) {
    // TODO: Unclear if the doubling from using SimpleConstraints will make this a problem
    if (basis.dimensions() == indexVars.size()) {
      break;
    }
    basis.addEquation(constraint.poly);
  }
  const IndexAccess& forms = basis.basis();
  if (forms.size() < indexVars.size()) {
    throw std::runtime_error("Underspecified set of equations in index variables");
  }

  // Print out equations
  if (VLOG_IS_ON(3)) {
    VLOG(3) << "In reduce, intial equations are";
    for (size_t i = 0; i < forms.size(); i++) {
      VLOG(3) << "  v" << i << " == " << forms[i].toString();
    }
  }

  // Convert to a matrix form + invert
  size_t size = forms.size();
  math::Matrix matrix;
  std::tie(matrix, std::ignore) = FromPolynomials(forms);
  if (!matrix.invert()) {
    throw std::runtime_error("Attempt to solve indexing equations failed due to singular matrix");
  }

  // Now convert back to equations
  IndexAccess inverses;
  for (size_t i = 0; i < size; i++) {
    IndexPoly poly;
    for (size_t j = 0; j < size; j++) {
      poly += matrix(i, j) * IndexPoly(std::string("v") + std::to_string(j));
    }
    inverses.push_back(poly);
  }

  // Vectorize variables
  std::vector<std::string> vec_idx(indexVars.begin(), indexVars.end());

  // Print new equations
  if (VLOG_IS_ON(3)) {
    VLOG(3) << "In reduce, reduced equations are:";
    for (size_t i = 0; i < size; i++) {
      VLOG(3) << "  " << vec_idx[i] << " == " << to_string(inverses[i]);
    }
  }

  for (auto& access : accesses) {
    for (auto& poly : access) {
      poly = ConvertVariables(poly, vec_idx, inverses);
    }
  }

  for (auto& constraint : range_constraints.constraints) {
    constraint = RangeConstraint(ConvertVariables(constraint.poly, vec_idx, inverses), constraint.range);
  }
}

void Contraction::Defractionalize() {
  IndexAccess polys;
  bool has_fract = false;
  std::set<std::string> vars;
  // TODO: This probably needs some attention for the conversion to SimpleConstraints
  for (const auto& constraint : range_constraints.constraints) {
    for (const auto& [key, value] : constraint.poly.getMap()) {
      if (denominator(value) != 1) {
        has_fract = true;
      }
      if (key != "") {
        vars.insert(key);
      }
    }
    polys.push_back(constraint.poly);
  }
  if (!has_fract) {
    IVLOG(3, "No fractions to Defract")
    return;
  }
  std::vector<std::string> vvars(vars.begin(), vars.end());

  math::Matrix mat;
  math::Vector vec;
  bool transform_constant = false;
  std::tie(mat, vec) = FromPolynomials(polys);
  IVLOG(3, "Original Matrix: " << mat);
  IVLOG(3, "Original Vector: " << vec);
  for (auto& v_entry : vec) {
    if (denominator(v_entry) != 1) {
      // Transform constant -> constant * dummy_var with 1 <= dummy_var < 2
      transform_constant = true;
      IVLOG(3, "Non-integer offset vector " << vec << " in defractionalization. Transforming constant to variable.");
      vvars.push_back("");
      mat.resize(mat.size1() + 1, mat.size2() + 1, true);
      // NOTE: For some reason, using boost::numeric::ublas::row() breaks debug builds.
      // So instead we write the equivalent code manually.
      for (size_t c = 0; c < mat.size2(); c++) {
        mat(mat.size1() - 1, c) = 0;
      }
      vec.resize(vec.size() + 1, true);
      // NOTE: For some reason, using boost::numeric::ublas::column() breaks debug builds.
      // So instead we write the equivalent code manually.
      for (size_t r = 0; r < mat.size1(); r++) {
        mat(r, mat.size2() - 1) = vec[r];
      }
      mat(mat.size1() - 1, mat.size2() - 1) = 1;
      break;
    }
  }

  if (!HermiteNormalForm(mat)) {
    throw std::runtime_error("Unable to perform Hermite Reduction during defractionalization");
  }
  IVLOG(4, "Matrix: " << mat);

  math::Matrix p = prod(trans(mat), mat);
  IVLOG(4, "Product Matrix: " << p);
  if (!p.invert()) {
    throw std::runtime_error("Unable to invert Hermite Matrix");
  }
  IVLOG(4, "Inverse Product Matrix: " << p);
  math::Matrix d = prod(mat, p);
  IVLOG(4, "Dual Matrix: " << d);
  IVLOG(3, "Normalized Dual Matrix: " << d);
  if (!HermiteNormalForm(d)) {
    throw std::runtime_error("Unable to perform Hermite Reduction on dual during defractionalization");
  }

  std::vector<RangeConstraint> new_cons;
  std::vector<std::map<Integer, IndexPoly>> mod_polys;
  for (size_t i = 0; i < d.size2(); i++) {
    IVLOG(4, "Computing splits for " << vvars[i]);
    std::set<Integer> splits;
    // For each element, compute the 'modulos' I need
    splits.insert(1);  // Sentinel
    for (size_t j = i + 1; j < d.size2(); j++) {
      Rational r = d(i, j) / d(j, j);
      splits.insert(denominator(r));
    }
    std::vector<Integer> split_vec(splits.begin(), splits.end());
    IVLOG(4, "List of splits: " << split_vec);
    // Now, break those into constraints + polynomials
    IndexPoly cpoly;
    std::map<Integer, IndexPoly> ipoly;
    for (size_t j = 0; j < split_vec.size(); j++) {
      std::string var = vvars[i] + "_" + std::to_string(j);
      // Add a constraint for non-final cases
      if (j < split_vec.size() - 1) {
        if (split_vec[j + 1] % split_vec[j] != 0) {
          throw std::runtime_error("Unable to remove modulo operations during defractionalization");
        }
        Integer div = split_vec[j + 1] / split_vec[j];
        new_cons.emplace_back(IndexPoly(var), static_cast<int64_t>(div));
      }
      // Add an entry to the polynomial
      cpoly += split_vec[j] * IndexPoly(var);
      // Add the polynomial to a lookup table
      Integer modof = (j + 1 < split_vec.size() ? split_vec[j + 1] : 0);
      ipoly[modof] = cpoly;
    }
    IVLOG(4, "List of polys: " << ipoly);
    mod_polys.push_back(ipoly);
  }

  // Now, make replacements
  std::map<std::string, IndexPoly> replacements;
  for (size_t i = 0; i < d.size2(); i++) {
    IndexPoly poly = d(i, i) * mod_polys[i][0];
    for (size_t j = 0; j < i; j++) {
      poly += d(j, i) * mod_polys[j][denominator(d(j, i) / d(i, i))];
    }
    replacements[vvars[i]] = poly;
  }

  IVLOG(3, "Replacements = " << replacements);
  IVLOG(3, "New Constraints = " << new_cons);

  for (auto& access : accesses) {
    for (auto& poly : access) {
      poly = ConvertPoly(poly, replacements, transform_constant);
    }
  }

  // TODO: Switch to SimpleConstraint
  for (auto& constraint : range_constraints.constraints) {
    constraint = RangeConstraint(ConvertPoly(constraint.poly, replacements, transform_constant), constraint.range);
  }
  if (transform_constant) {
    // Add constraint for the constant term if it's being transformed
    range_constraints.constraints.emplace_back(
        RangeConstraint(ConvertPoly(IndexPoly(1), replacements, transform_constant) - 1, 1));
  }
  for (const auto& constraint : new_cons) {
    range_constraints.constraints.push_back(constraint);
  }
}

bool Contraction::NeedReduce() const {
  for (const auto& poly : accesses[0]) {
    if (poly.getMap().size() > 2 || (poly.getMap().size() == 2 && poly.constant() == 0)) {
      return true;
    }
  }
  return false;
}

void Contraction::DeduceRangeConstraints() {
  ConstrainIndexVarsToInts();
  // `unmerged` will track SimpleConstraints that we have yet to merge into a RangeConstraint.
  // Each entry is a collection of unpaired simple constraints, all parallel and pointing in the same direction
  std::list<std::list<SimpleConstraint>> unmerged;
  for (const auto& cons : constraints) {
    bool has_matched = false;
    // First try to merge with an existing RangeConstraint
    for (auto& r_cons : range_constraints.constraints) {
      if (cons.poly.tryDivide(r_cons.poly, true)) {
        r_cons = IntersectParallelConstraintPair(r_cons, cons);
        has_matched = true;
        break;
      }
    }
    if (has_matched) {
      continue;
    }
    // Then try to merge with another unpaired SimpleConstraint
    for (auto cons_set = unmerged.begin(); cons_set != unmerged.end(); cons_set++) {
      // Iterate through the collections of "parallel same direction" constraints, see if `cons` is parallel to any of
      // them
      auto ratio = cons.poly.tryDivide(cons_set->begin()->poly, true);
      if (ratio > 0) {
        // Parallel in same direction:
        // Group with this set
        cons_set->emplace_back(cons);
        has_matched = true;
        break;
      }
      if (ratio < 0) {
        // Parallel in opposite direction:
        // Merge with everything in this set to create RangeConstraint that intersects all
        auto r_cons = IntersectOpposedSimpleConstraints(cons, *cons_set->begin());
        cons_set->erase(cons_set->begin());
        for (auto& other_cons : *cons_set) {
          r_cons = IntersectParallelConstraintPair(r_cons, other_cons);
        }
        range_constraints.constraints.push_back(r_cons);
        unmerged.erase(cons_set);
        has_matched = true;
        break;
      }
      // ratio == 0, i.e. not parallel
    }
    if (!has_matched) {
      // This is not parallel to anything, move to new set in unmerged
      unmerged.emplace_back(std::list<SimpleConstraint>{cons});
    }
  }
  if (!unmerged.empty()) {
    throw std::runtime_error("Unable to pair all constraints");
    // TODO: Switch to better log:
    // throw std::runtime_error("Unable to pair all constraints, started with " + to_string(constraints)
    //   + ", unable to match " + to_string(unmerged));
  }
}

BoundsAndConstraints Contraction::ComputeBounds(llvm::ArrayRef<stripe::TensorType> shapes, bool no_reduce) {
  // Because we construct `range_constraints` from `constraints` and then ignore the information in `constraints` in
  // favor of `range_constraints`, this section is a bit brittle. Check assumptions about whether `constraints` or
  // `range_constraints` are used when working with this code.
  DeduceRangeConstraints();
  GatherConstraints(shapes);
  IVLOG(3, "Constraints:" << to_string(range_constraints.constraints));
  // Reduce if needed
  if (NeedReduce() && !no_reduce) {
    ReduceOutputPolynomials();
    GatherConstraints(shapes);
  }
  range_constraints.MergeParallelConstraints();
  IVLOG(3, "Merged Parallel Constraints:" << to_string(range_constraints.constraints));
  // Defract if needed (defract does early return if not required)
  Defractionalize();
  // Gather the constraints from index bounds
  GatherConstraints(shapes);
  // New parallel constraints might have been introduced by defract; re-merge them
  range_constraints.MergeParallelConstraints();
  return range_constraints.ComputeBounds();
}

math::Affine Integerize(const IndexPoly& poly, const IndexBounds& bounds) {
  math::Affine result;
  for (const auto& term : poly.getMap()) {
    if (denominator(term.second) != 1) {
      throw std::runtime_error("Non-integer polynomial in Integerize");
    }
    auto int_value = static_cast<int64_t>(numerator(term.second));
    if (term.first.empty()) {
      result += int_value;
    } else {
      const auto& bound = bounds.at(term.first);
      result += int_value * bound.min;
      result += math::Affine(term.first, int_value);
    }
  }
  return result;
}

}  // namespace pmlc::dialect::tile
