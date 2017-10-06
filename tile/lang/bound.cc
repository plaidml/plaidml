#include "tile/lang/bound.h"
#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>

namespace vertexai {
namespace tile {
namespace lang {

Contraction ConstrainIndexVarsToInts(const Contraction& c) {
  Contraction c_new = c;

  std::set<std::string> varsUsed;
  varsUsed = c.getIndexVariables();

  for (const auto& variable : varsUsed) {
    if (!variable.empty()) {  // Constant components not used
      const int32_t constraintBoundWidth = 1000000000;
      c_new.constraints.push_back(
          SymbolicConstraint(RangeConstraint(Polynomial(variable) + constraintBoundWidth / 2, constraintBoundWidth)));
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
        printstring("Shape mismatch during contraint gathering: %zu vs %zu", shapes.size(), c.specs.size()));
  }
  // For each shape...
  for (size_t i = 0; i < c.specs.size(); i++) {
    const IndexSpec& spec = c.specs[i].spec;
    // Sanity check the dimensions
    if (spec.size() != shapes[i].dims.size()) {
      throw std::runtime_error(printstring("More indexes than dimensions for tensor: %zu:%s %zu > %zu", i,
                                           c.specs[i].id.c_str(), spec.size(), shapes[i].dims.size()));
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

bool IsImplied(const SimpleConstraint& c, const IndexBounds& b) {
  const Polynomial& p = c.poly;
  Rational worst = p.constant();
  for (const auto& kvp : p.getMap()) {
    if (kvp.first == "") {
      continue;
    }
    if (kvp.second < 0) {
      worst += kvp.second * b.find(kvp.first)->second.min;
    } else {
      worst += kvp.second * b.find(kvp.first)->second.max;
    }
  }
  return (worst <= c.rhs);
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

RangeConstraint IntersectParallelConstraintPair(const RangeConstraint& constraint1,
                                                const RangeConstraint& constraint2) {
  // Combines two parallel constraints into one. See merge-parallel.tex in
  // /tile/lang for more details.

  // ratio is only used to check for invalid input, but if we don't check for
  // garbage input, then IntersectParallelConstraintPair will in many cases give
  // us garbage output with no errors or warnings. That seems bad enough to be
  // worth computing ratio here.
  Rational ratio = constraint1.poly.tryDivide(constraint2.poly, true);
  if (ratio == 0) {
    throw std::invalid_argument("Error: Parameters of IntersectParallelConstraintPair must be parallel");
  }

  std::string NonzeroIndex = constraint1.poly.GetNonzeroIndex();
  if (NonzeroIndex.empty() || constraint2.poly.GetNonzeroIndex().empty()) {
    throw std::invalid_argument(std::string("Error: Constant polynomial passed to IntersectParallelConstraintPair.\n") +
                                "This should have been eliminated in MergeParallelConstraints.");
  }
  Rational coeff1 = constraint1.poly[NonzeroIndex];
  Rational coeff2 = constraint2.poly[NonzeroIndex];
  Rational c1ratio = GCD(coeff1, coeff2) / coeff1;
  Rational c2ratio = GCD(coeff1, coeff2) / coeff2;
  Integer c1IntegerFactor = getIntegerFactorFromRatio(c1ratio);
  Integer c2IntegerFactor = getIntegerFactorFromRatio(c2ratio);

  Polynomial PartialConstraint(c1ratio * constraint1.poly);

  // We need i, j such that i * c2IntFact - j * c1IntFact = gcd of IntFacts
  Integer i;
  Integer j;
  Integer GCDofIntFactors = XGCD(c1IntegerFactor, c2IntegerFactor, i, j);
  if (GCDofIntFactors != 1 && GCDofIntFactors != -1) {
    throw std::logic_error("Error: In IntersectParallelConstraintPair the integer factors must be relatively prime.");
  }
  Rational offsetMergeConstantRational = c1IntegerFactor * c2IntegerFactor * constraint2.poly.constant() -
                                         c1IntegerFactor * c2IntegerFactor * constraint1.poly.constant();
  Integer offsetMergeConstant;
  if (denominator(offsetMergeConstantRational) == -1) {
    offsetMergeConstant = -numerator(offsetMergeConstantRational);
  } else if (denominator(offsetMergeConstantRational) == 1) {
    offsetMergeConstant = numerator(offsetMergeConstantRational);
  } else {
    // Throw indicating the constraints are mutually exclusive
    std::stringstream ErrorMessage;
    ErrorMessage << "Error: IntersectParallelConstraintPair does not yet handle mutually exclusive constraints."
                 << "\nConstraint 1 poly: " << constraint1.poly.toString()
                 << "\nConstraint 1 range: " << std::to_string(constraint1.range)
                 << "\nConstraint 2 poly: " << constraint2.poly.toString()
                 << "\nConstraint 2 range: " << std::to_string(constraint2.range);
    throw std::invalid_argument(ErrorMessage.str());
  }
  // Transform i and j into the actual solutions of the linear Diophantine equation k2 i - k1 j = b
  i *= GCDofIntFactors * offsetMergeConstant;
  j *= GCDofIntFactors * offsetMergeConstant;

  Rational FractionalOffset =
      constraint1.poly.constant() +
      i * c1ratio;  // Note that this is only correct up to an integer constant (which we compute below)
  Integer IntegralOffset1 =
      (c1IntegerFactor > 0)
          ? Ceil(c1ratio * constraint1.poly.constant() - FractionalOffset)
          : Floor(c1ratio * constraint1.poly.constant() - c1ratio * constraint1.range - FractionalOffset + 1);
  Integer IntegralOffset2 =
      (c2IntegerFactor > 0)
          ? Ceil(c2ratio * constraint2.poly.constant() - FractionalOffset)
          : Floor(c2ratio * constraint2.poly.constant() - c2ratio * constraint2.range - FractionalOffset + 1);
  Rational NetOffset = FractionalOffset + Min(IntegralOffset1, IntegralOffset2);
  Integer Range1 = (c1IntegerFactor > 0)
                       ? Ceil(c1ratio * constraint1.range - c1ratio * constraint1.poly.constant() + NetOffset)
                       : Floor(-c1ratio * constraint1.poly.constant() + NetOffset + 1);
  Integer Range2 = (c2IntegerFactor > 0)
                       ? Ceil(c2ratio * constraint2.range - c2ratio * constraint2.poly.constant() + NetOffset)
                       : Floor(-c2ratio * constraint2.poly.constant() + NetOffset + 1);
  // I'm not sure but it looks like boost doesn't throw when an explicit narrowing
  // conversion turns out to be lossy, so we'll check this ourselves
  if (Range1 > INT64_MAX && Range2 > INT64_MAX)
    throw std::out_of_range("Error: Bound range in IntersectParallelConstraintPair overflows int64.");
  int64_t Range = (int64_t)(Min(Range1, Range2));

  PartialConstraint.setConstant(NetOffset);

  return RangeConstraint(PartialConstraint, Range);
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
  std::vector<Polynomial> objectives;
  for (const std::string& var : variableNames) {
    objectives.emplace_back(var);
    objectives.emplace_back(var, -1);
  }
  std::map<Polynomial, bilp::ILPResult> result = solver.batch_solve(constraints, objectives);
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
