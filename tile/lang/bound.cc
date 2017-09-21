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

class CoinSilencer : public CoinMessageHandler {
  // The whole point of this class is to NOT print, so just return on print()
  virtual int print() { return 0; }
};

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
  std::sort(out.begin(), out.end(),
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

// TODO(T133): Check size of LP-Solve problem to prevent exponential slowdown
static std::pair<CoinPackedMatrix, std::vector<std::string>> ConstraintsToSparseMatrix(
    const std::vector<RangeConstraint>& constraints) {
  // Transforms constraints into a sparse matrix in a format Coin can comprehend.
  // Also reports which variable indices go with which variable names

  // Each variable is associated with a column in the output matrix. colNumberMap
  // lets us get this column's index from the variable name.
  std::map<std::string, int> colNumberMap;
  int variables_in_model = 0;  // Total # variables among all constraints

  // Matrix is encoded by triples (rowIdx, colIdx, coeff) -- so that
  // <Matrix>[rowIdx, colIdx] = coeff
  // entries stores these in the order (rowIdx, colIdx, coeff)
  std::vector<std::tuple<int, int, Rational>> entries;

  for (int rowNumber = 0; rowNumber < constraints.size(); ++rowNumber) {
    const auto& c = constraints[rowNumber];
    for (const auto& kvp : c.poly.getMap()) {
      if (kvp.first != "") {  // Constant terms don't go into this matrix
        // Get the int index of this variable, or give it one if we haven't seen it before
        auto it = colNumberMap.find(kvp.first);
        if (it == colNumberMap.end()) {
          it = colNumberMap.insert(std::map<std::string, int>::value_type(kvp.first, variables_in_model)).first;
          variables_in_model++;
        }
        entries.push_back(std::tuple<int, int, double>(rowNumber, it->second, static_cast<double>(kvp.second)));
      }
    }
  }

  // Read out entries into three vectors for Coin to read
  std::vector<int> rowIndices;
  std::vector<int> colIndices;
  std::vector<double> coefficients;
  for (const auto& entry : entries) {
    rowIndices.push_back(std::get<0>(entry));
    colIndices.push_back(std::get<1>(entry));
    coefficients.push_back(static_cast<double>(std::get<2>(entry)));
  }
  if (VLOG_IS_ON(5)) {
    std::ostringstream msg;
    msg << "Handing to COIN:";
    msg << "\nrowIndices:";
    for (const auto& index : rowIndices) {
      msg << " " << std::to_string(index);
    }
    msg << "\ncolIndices:";
    for (const auto& index : colIndices) {
      msg << " " << std::to_string(index);
    }
    msg << "\ncoefficients:";
    for (const auto& coeff : coefficients) {
      msg << " " << std::to_string(coeff);
    }
    IVLOG(5, msg.str());
  }

  std::vector<std::string> varNames(entries.size());
  for (const auto& kvp : colNumberMap) {
    varNames[kvp.second] = kvp.first;
  }
  if (VLOG_IS_ON(5)) {
    std::ostringstream msg;
    msg << "Var names:";
    for (const auto& name : varNames) {
      msg << " " << name;
    }
    IVLOG(5, msg.str());
  }

  return std::pair<CoinPackedMatrix, std::vector<std::string>>(
      CoinPackedMatrix(false, &rowIndices[0], &colIndices[0], &coefficients[0], entries.size()), varNames);
}

// TODO(T133): Check size of integer programming problem to prevent exponential slowdown
std::tuple<IndexBounds, std::vector<SimpleConstraint>> ComputeBounds(const std::vector<RangeConstraint>& constraints) {
  Integer lcm = 1;
  for (int rowNumber = 0; rowNumber < constraints.size(); ++rowNumber) {
    const auto& c = constraints[rowNumber];
    for (const auto& kvp : c.poly.getMap()) {
      lcm = LCM(lcm, denominator(kvp.second));
    }
  }

  // Create the constraint matrix to pass to the model (also LCM for later use
  // is computed now to reduce iterations over constraints)
  auto sparseMatAndVarNames = ConstraintsToSparseMatrix(constraints);
  CoinPackedMatrix constraintMatrix = sparseMatAndVarNames.first;
  std::vector<std::string> variableNames = sparseMatAndVarNames.second;

  if (VLOG_IS_ON(5)) {
    size_t num_elems = constraintMatrix.getNumElements();
    const double* elems = constraintMatrix.getElements();
    const int* idxs = constraintMatrix.getIndices();
    std::ostringstream msg;
    msg << "Elements:";
    for (size_t i = 0; i < num_elems; ++i) {
      msg << " " << elems[i];
    }
    msg << '\n';
    msg << "Indices:";
    for (size_t i = 0; i < num_elems; ++i) {
      msg << " " << idxs[i];
    }
    IVLOG(5, msg.str());
  }

  // Write the bounds for the constraints and the variables
  std::vector<double> constraintLowerBounds;
  std::vector<double> constraintUpperBounds;
  std::vector<double> variableLowerBounds;
  std::vector<double> variableUpperBounds;
  for (const auto& c : constraints) {
    constraintLowerBounds.push_back(static_cast<double>(-c.poly.constant()));
    constraintUpperBounds.push_back(static_cast<double>(c.range - 1 - c.poly.constant()));
  }
  for (int i = 0; i < constraintMatrix.getNumCols(); ++i) {
    variableLowerBounds.push_back(-10e9);
    variableUpperBounds.push_back(10e9);
  }
  if (VLOG_IS_ON(5)) {
    std::ostringstream msg;
    msg << "Constraint Bounds: ";
    for (int i = 0; i < constraints.size(); ++i) {
      msg << constraintLowerBounds[i] << " <= (" << i << ") <= " << constraintUpperBounds[i];
      if (i != constraints.size() - 1) msg << ", ";
    }
    IVLOG(5, msg.str());
  }

  // Run the solver for each variable min + max
  IndexBounds out;
  for (size_t i = 0; i < constraintMatrix.getNumCols(); ++i) {
    Bound result;
    std::vector<double> objective(constraintMatrix.getNumCols(), 0);

    std::vector<int> MinOrMax = {-1, 1};  // -1 encodes maximize, 1 encodes minimize
    for (const auto& objective_type : MinOrMax) {
      // Make a Clp model
      OsiClpSolverInterface model;

      // Set CLP log verbosity
      CoinSilencer silentHandler;
      if (VLOG_IS_ON(4)) {
        model.setLogLevel(2);
        model.messageHandler()->setLogLevel(2);
      } else if (VLOG_IS_ON(5)) {
        model.setLogLevel(4);
        model.messageHandler()->setLogLevel(4);
      } else {
        model.passInMessageHandler(&silentHandler);
      }

      objective[i] = objective_type;
      model.loadProblem(constraintMatrix, &variableLowerBounds[0], &variableUpperBounds[0], &objective[0],
                        &constraintLowerBounds[0], &constraintUpperBounds[0]);
      if (i == 0) {
        for (int j = 0; j < constraintMatrix.getNumCols(); ++j) {
          model.setInteger(j);
        }
      }
      model.branchAndBound();

      if (VLOG_IS_ON(5)) {
        std::ostringstream solnTerminationInfo;
        solnTerminationInfo << "Integer Program Model (i=" << std::to_string(i) << ") "
                            << "... isAbandoned: " << model.isAbandoned() << "\n"
                            << "... isProvenOptimal: " << model.isProvenOptimal() << "\n"
                            << "... isProvenPrimalInfeasible: " << model.isProvenPrimalInfeasible() << "\n"
                            << "... isProvenDualInfeasible: " << model.isProvenDualInfeasible() << "\n"
                            << "... isPrimalObjectiveLimitReached: " << model.isPrimalObjectiveLimitReached() << "\n"
                            << "... isDualObjectiveLimitReached: " << model.isDualObjectiveLimitReached() << "\n"
                            << "... isIterationLimitReached: " << model.isIterationLimitReached();
        IVLOG(5, solnTerminationInfo.str());
      }
      if (model.isProvenPrimalInfeasible()) {
        std::string msg = "vertexai::tile::lang::ComputeBounds: Constraints are infeasible; nothing to compute";
        throw std::runtime_error(msg);
      } else if (model.isProvenDualInfeasible()) {
        std::string msg = "vertexai::tile::lang::ComputeBounds: Constraints unbounded or infeasible; aborting";
        throw std::runtime_error(msg);
      } else if (model.isAbandoned()) {
        std::string msg = "vertexai::tile::lang::ComputeBounds: Model numerically ill-behaved; aborting";
        throw std::runtime_error(msg);
      }
      if (!model.isProvenOptimal()) {
        throw std::runtime_error(
            "vertexai::tile::lang::ComputeBounds: Unable to prove bounds from linear program are optimal.");
      }
      double solution_buffer = model.getObjValue();
      Rational rationalized_solution;
      if (solution_buffer > 0) {
        rationalized_solution = Rational(Integer(int64_t(solution_buffer + 0.01)), lcm);
      } else {
        rationalized_solution = Rational(Integer(int64_t(solution_buffer - 0.01)), lcm);
      }
      int64_t final_result_buffer = static_cast<int64_t>(Floor(rationalized_solution));
      if (objective_type == -1) {
        result.max = -final_result_buffer;
      } else {
        result.min = final_result_buffer;
      }
    }
    out[variableNames[i]] = result;
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
