#pragma once

#include <gtest/gtest_prod.h>
#include <map>
#include <string>
#include <vector>

#include "tile/lang/bignum.h"
#include "tile/lang/milp/tableau.h"

namespace vertexai {
namespace tile {
namespace lang {
namespace milp {

enum class BranchMethod { full, face, open };

struct ILPResult {
  Polynomial obj;
  Rational obj_val;
  std::map<std::string, Rational> soln;
  ILPResult(Polynomial objective, Rational objective_val, std::map<std::string, Rational> solution)
      : obj{objective}, obj_val{objective_val}, soln{solution} {};
};

class ILPSolver {
 public:
  // TODO(T1146): Unify solve and batch_solve reporting methods
  // Finds minimal value of objective subject to constraints (and subject to
  // every variable and every constraint value being an integer)
  bool solve(const std::vector<RangeConstraint>& constraints, const Polynomial objective);
  // Solves a tableau representing an ILP problem
  bool solve(Tableau& tableau, bool already_canonical = false);  // NOLINT(runtime/references)
  // Returns a solution for each objective, all subject to the same constraints
  std::vector<ILPResult> batch_solve(const std::vector<RangeConstraint>& constraints,
                                     const std::vector<Polynomial>& objectives);
  // Reports the minimized value of the objective for last solved problem
  Rational reportObjective() const { return best_objective; }
  // Reports the variable values minimizing the objective for last solved problem
  std::map<std::string, Rational> reportSolution() const;
  // Reports the variable values in var_names_ order for the last solved problem
  std::vector<Rational> getSymbolicSolution() const { return best_solution; }

  // Returns the solver to a neutral state to prepare to solve another problem
  void clean();

 private:
  FRIEND_TEST(MilpTest, BasicTableauTest);
  FRIEND_TEST(MilpTest, SimpleOptimizeTest);
  FRIEND_TEST(MilpTest, OptimizeTest2D);
  FRIEND_TEST(MilpTest, TrivialILPTest);
  bool feasible_found = false;
  Rational best_objective = 0;
  std::vector<Rational> best_solution;
  std::vector<std::string> var_names_;

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
}  // namespace milp
}  // namespace lang
}  // namespace tile
}  // namespace vertexai
