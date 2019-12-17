#include "tile/lang/bound.h"

#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>

#include <boost/format.hpp>

namespace vertexai {
namespace tile {
namespace lang {

using namespace math;  // NOLINT

Contraction ConstrainIndexVarsToInts(const Contraction& c) {
  Contraction c_new = c;

  std::set<std::string> varsUsed;
  varsUsed = c.getIndexVariables();

  for (const auto& variable : varsUsed) {
    if (!variable.empty()) {  // Constant components not used
      const int32_t constraintBoundWidth = 1000000000;
      c_new.constraints.push_back(SymbolicConstraint(
          RangeConstraint(Polynomial<Rational>(variable) + constraintBoundWidth / 2, constraintBoundWidth)));
    }
  }

  return c_new;
}

static Integer getIntegerFactorFromRatio(Rational ratio) {
  // Computes 1/ratio, but with checks to ensure that ratio is of the form 1/k
  // for some nonzero integer k and to confirm that this k is what is returned
  if (numerator(ratio) != 1) {
    if (numerator(ratio) == -1) {
      return -denominator(ratio);
    } else {
      // If GCD and Rational work like they should, this can't happen,
      // since numerator is factor of denominator
      throw std::logic_error("Error: IntersectParallelConstraintPair expects constraint ratios to be of the form 1/k.");
    }
  }
  return denominator(ratio);
}

std::vector<RangeConstraint> GatherConstraints(const Contraction& c, const std::vector<TensorShape>& shapes) {
  // Make the output collection
  std::vector<RangeConstraint> out;
  // Add all the simple constraints
  for (const auto& cons : c.constraints) {
    out.push_back(cons.bound);
  }
  // Sanity check the shapes
  if (shapes.size() != c.specs.size()) {
    throw std::runtime_error(
        str(boost::format("Shape mismatch during contraint gathering: %zu vs %zu") % shapes.size() % c.specs.size()));
  }
  // For each shape...
  for (size_t i = 0; i < c.specs.size(); i++) {
    const IndexSpec& spec = c.specs[i].spec;
    // Sanity check the dimensions
    if (spec.size() != shapes[i].dims.size()) {
      throw std::runtime_error(str(boost::format("More indexes than dimensions for tensor: %zu:%s %zu > %zu") %  //
                                   i % c.specs[i].id % spec.size() % shapes[i].dims.size()));
    }
    // For each dimension...
    for (size_t j = 0; j < spec.size(); j++) {
      // Add a new constraint
      out.push_back(RangeConstraint(spec[j], shapes[i].dims[j].size));
    }
  }
  std::stable_sort(out.begin(), out.end(),
                   [](const RangeConstraint& c1, const RangeConstraint& c2) { return (c1.range < c2.range); });
  // Return the output
  return out;
}

void MergeParallelConstraints(std::vector<RangeConstraint>* constraints) {
  auto i = constraints->begin();
  while (i != constraints->end()) {
    // Remove trivially true constraints, fail for trivially false constraints
    // By trivially true/false, I mean constraints of the form 0 <= c < r,
    // where c is a number rather than a polynomial.
    if (i->poly.GetNonzeroIndex().empty()) {
      if (0 <= i->poly.constant() && i->poly.constant() < i->range) {
        // Constraint is trivially true; remove and continue
        if (i == constraints->begin()) {
          constraints->erase(i);
          i = constraints->begin();
          continue;
        } else {  // i-- exists
          i--;
          constraints->erase(i + 1);
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
    for (auto j = i + 1; j != constraints->end(); ++j) {
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
        constraints->erase(j + 1);
      }
    }
    ++i;
  }
}

Rational UnifiedOffset(const Rational& c1, const Rational& c2, const Integer& n1, const Integer& n2) {
  std::set<Rational> offsets;
  if (n1 > std::numeric_limits<std::size_t>::max() || n2 > std::numeric_limits<std::size_t>::max()) {
    throw std::out_of_range("Cannot unify offset when relative quotient exceeds size_t.");
  }
  for (size_t i = 0; i < Abs(n1); ++i) {
    offsets.insert(std::end(offsets), FracPart((c1 + i) / n1));
  }
  for (size_t j = 0; j < Abs(n2); ++j) {
    Rational offset = FracPart((c2 + j) / n2);
    if (offsets.count(offset)) {
      return offset;
    }
  }
  IVLOG(1, "Failed to compute UnifiedOffset(" << c1 << ", " << c2 << ", " << n1 << ", " << n2 << ").");
  throw std::runtime_error("Merging constraints with empty intersection.");
}

RangeConstraint IntersectParallelConstraintPair(const RangeConstraint& constraint1,
                                                const RangeConstraint& constraint2) {
  // Combines two parallel constraints into one. See merge-parallel.tex in
  // /tile/lang for more details.
  IVLOG(5, "Merging the parallel constraints " << constraint1 << ", " << constraint2);
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
  Rational q1_low = Min(-c1 / n1, (constraint1.range - 1 - c1) / n1);
  Rational q1_hi = Max(-c1 / n1, (constraint1.range - 1 - c1) / n1);
  Rational q2_low = Min(-c2 / n2, (constraint2.range - 1 - c2) / n2);
  Rational q2_hi = Max(-c2 / n2, (constraint2.range - 1 - c2) / n2);
  Integer lower_bound = Max(Ceil(q1_low + d), Ceil(q2_low + d));
  Integer upper_bound = Min(Floor(q1_hi + d), Floor(q2_hi + d));
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
  Polynomial<Rational> p(constraint1.poly / n1);
  p.setConstant(merged_offset);
  return RangeConstraint(p, r);
}

static std::set<std::string> variablesUsed(const std::vector<RangeConstraint>& constraints) {
  // Returns all the variables appearing in the constraints
  std::set<std::string> ret;

  for (const RangeConstraint& c : constraints) {
    for (const auto& kvp : c.poly.getMap()) {
      const std::string& key = kvp.first;
      if (key != "") {  // Do nothing for constant term
        ret.emplace(key);
      }
    }
  }

  return ret;
}

// TODO(T133): Check size of integer programming problem to prevent slowdown
std::tuple<IndexBounds, std::vector<SimpleConstraint>> ComputeBounds(const std::vector<RangeConstraint>& constraints) {
  std::set<std::string> variableNames = variablesUsed(constraints);

  // Run the solver for each variable min + max
  bilp::ILPSolver solver;
  IndexBounds out;
  std::vector<Polynomial<Rational>> objectives;
  for (const std::string& var : variableNames) {
    objectives.emplace_back(var);
    objectives.emplace_back(var, -1);
  }
  std::map<Polynomial<Rational>, bilp::ILPResult> result = solver.batch_solve(constraints, objectives);
  for (const auto& kvp : result) {
    // ILPResult lists the objective for each requested optimization. Since we
    // used a monomial for each objective, GetNonzeroIndex returns the name of
    // the variable. Then we grab its coefficient to see if we were requesting
    // minimization or maximization
    std::string var = kvp.first.GetNonzeroIndex();
    if (kvp.first[var] == 1) {
      out[var].min = static_cast<int64_t>(kvp.second.obj_val);
    } else if (kvp.first[var] == -1) {
      out[var].max = static_cast<int64_t>(-kvp.second.obj_val);
    } else {
      throw std::runtime_error("Internal error: unexpected ILP objective type");
    }
  }

  // Remove constraints which are implied
  std::vector<SimpleConstraint> remaining;
  for (const RangeConstraint& c : constraints) {
    if (!IsImplied(c.lowerBound(), out)) {
      remaining.push_back(c.lowerBound());
    }
    if (!IsImplied(c.upperBound(), out)) {
      remaining.push_back(c.upperBound());
    }
  }

  return std::tie(out, remaining);
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
