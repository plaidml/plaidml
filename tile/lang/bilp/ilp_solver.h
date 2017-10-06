#pragma once

#include <algorithm>
#include <gtest/gtest_prod.h>
#include <map>
#include <string>
#include <vector>

#include "tile/lang/bignum.h"
#include "tile/lang/bilp/tableau.h"

namespace vertexai {
namespace tile {
namespace lang {
namespace bilp {

enum class BranchMethod { full, face, open };

struct ILPResult {
  Rational obj_val;
  std::map<std::string, Rational> soln;
  ILPResult() {}
  ILPResult(Rational objective_val, std::map<std::string, Rational> solution)
      : obj_val{objective_val}, soln{solution} {};
};

class ILPSolver {
 public:
  // Finds minimal value of objective subject to constraints (and subject to
  // every variable and every constraint value being an integer)
  ILPResult solve(const std::vector<RangeConstraint>& constraints, const Polynomial objective);
  // Returns a solution for each objective, all subject to the same constraints
  std::map<Polynomial, ILPResult> batch_solve(const std::vector<RangeConstraint>& constraints,
                                              const std::vector<Polynomial>& objectives);

 private:
  FRIEND_TEST(BilpTest, BasicTableauTest);
  FRIEND_TEST(BilpTest, SimpleOptimizeTest);
  FRIEND_TEST(BilpTest, OptimizeTest2D);
  FRIEND_TEST(BilpTest, TrivialILPTest);
  bool feasible_found = false;
  Rational best_objective = 0;
  std::vector<Rational> best_solution;
  std::vector<std::string> var_names_;

  // Solves a tableau representing an ILP problem
  ILPResult solve(Tableau& tableau, bool already_canonical = false);  // NOLINT(runtime/references)
  // Reports the minimized value of the objective for last solved problem
  Rational reportObjective() const { return best_objective; }
  // Reports the variable values minimizing the objective for last solved problem
  std::map<std::string, Rational> reportSolution() const;
  // Reports the variable values in var_names_ order for the last solved problem
  std::vector<Rational> getSymbolicSolution() const { return best_solution; }

  // Returns the solver to a neutral state to prepare to solve another problem
  void clean();

  // Transform constraints & objective to the tableau representing them
  // Can omit objective to set up just the constraints; this tableau can then be
  // copied to solve multiple problems with the same constraints
  Tableau makeStandardFormTableau(const std::vector<RangeConstraint>& constraints,
                                  const Polynomial objective = Polynomial());
  // Add an additional constraint that reduces the real feasible region but that
  // leaves the integral feasible region unchanged
  Tableau addGomoryCut(const Tableau& t, size_t row);

  // Perform one step of the solve algorithm (i.e. either find an integer feasible
  // solution, or find a noninteger feasible solution, add a cut, and iterate)
  void solve_step(Tableau& tableau, bool already_canonical = false);  // NOLINT(runtime/references)
};
}  // namespace bilp
}  // namespace lang
}  // namespace tile
}  // namespace vertexai
