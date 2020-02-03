#pragma once

#include <gtest/gtest_prod.h>

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "pmlc/util/bilp/tableau.h"
#include "pmlc/util/math/bignum.h"

namespace pmlc::util::bilp {

enum class BranchMethod { full, face, open };

struct ILPResult {
  math::Rational obj_val;
  std::map<std::string, math::Rational> soln;
  ILPResult() {}
  ILPResult(math::Rational objective_val,
            std::map<std::string, math::Rational> solution)
      : obj_val{objective_val}, soln{solution} {};
};

class ILPSolver {
public:
  // Finds minimal value of objective subject to constraints (and subject to
  // every variable and every constraint value being an integer)
  ILPResult solve(const std::vector<math::RangeConstraint> &constraints,
                  const math::Polynomial<math::Rational> objective);
  ILPResult solve(const std::vector<math::SimpleConstraint> &constraints,
                  const math::Polynomial<math::Rational> objective);
  // Returns a solution for each objective, all subject to the same constraints
  std::map<math::Polynomial<math::Rational>, ILPResult>
  batch_solve(const std::vector<math::SimpleConstraint> &constraints,
              const std::vector<math::Polynomial<math::Rational>> &objectives);
  std::map<math::Polynomial<math::Rational>, ILPResult>
  batch_solve(const std::vector<math::RangeConstraint> &constraints,
              const std::vector<math::Polynomial<math::Rational>> &objectives);
  std::map<math::Polynomial<math::Rational>, ILPResult>
  batch_solve(Tableau *tableau,
              const std::vector<math::Polynomial<math::Rational>> &objectives);

  void set_throw_infeasible(bool b) { throw_infeasible = b; }

private:
  FRIEND_TEST(BilpTest, BasicTableauTest);
  FRIEND_TEST(BilpTest, SimpleOptimizeTest);
  FRIEND_TEST(BilpTest, OptimizeTest2D);
  FRIEND_TEST(BilpTest, TrivialILPTest);
  bool feasible_found = false;
  bool throw_infeasible = true;
  math::Rational best_objective = 0;
  std::vector<math::Rational> best_solution;
  std::vector<std::string> var_names_;

  // Solves a tableau representing an ILP problem
  ILPResult solve(Tableau &tableau,
                  bool already_canonical = false); // NOLINT(runtime/references)
  // Reports the minimized value of the objective for last solved problem
  math::Rational reportObjective() const { return best_objective; }
  // Reports the variable values minimizing the objective for last solved
  // problem
  std::map<std::string, math::Rational> reportSolution() const;
  // Reports the variable values in var_names_ order for the last solved problem
  std::vector<math::Rational> getSymbolicSolution() const {
    return best_solution;
  }

  // Returns the solver to a neutral state to prepare to solve another problem
  void clean();
  // Add an additional constraint that reduces the real feasible region but that
  // leaves the integral feasible region unchanged
  Tableau addGomoryCut(const Tableau &t, size_t row);

  // Perform one step of the solve algorithm (i.e. either find an integer
  // feasible solution, or find a noninteger feasible solution, add a cut, and
  // iterate)
  void solve_step(Tableau &tableau,
                  bool already_canonical = false); // NOLINT(runtime/references)
};

// Transform constraints & objective to the tableau representing them
// Can omit objective to set up just the constraints; this tableau can then be
// copied to solve multiple problems with the same constraints
Tableau
makeStandardFormTableau(const std::vector<math::RangeConstraint> &constraints,
                        const math::Polynomial<math::Rational> objective =
                            math::Polynomial<math::Rational>());

// Transform constraints & objective to the tableau representing them
// Can omit objective to set up just the constraints; this tableau can then be
// copied to solve multiple problems with the same constraints
// The simple constraints used here also assert that the constrained polynomial
// must evaluate to an integer
Tableau
makeStandardFormTableau(const std::vector<math::SimpleConstraint> &constraints,
                        const math::Polynomial<math::Rational> objective =
                            math::Polynomial<math::Rational>());

} // namespace pmlc::util::bilp
