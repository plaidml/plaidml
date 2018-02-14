#include "tile/lang/bilp/ilp_solver.h"

namespace vertexai {
namespace tile {
namespace lang {
namespace bilp {

std::map<std::string, Rational> ILPSolver::reportSolution() const {
  std::vector<Rational> sym_soln = getSymbolicSolution();
  std::map<std::string, Rational> soln;
  for (size_t i = 0; i < sym_soln.size(); ++i) {
    std::string var = var_names_[i];
    if (var.substr(var.length() - 4, 4) == "_pos") {
      soln[var.substr(1, var.length() - 5)] += sym_soln[i];
    } else if (var.substr(var.length() - 4, 4) == "_neg") {
      soln[var.substr(1, var.length() - 5)] -= sym_soln[i];
    }
  }
  return soln;
}

std::map<Polynomial, ILPResult> ILPSolver::batch_solve(const std::vector<RangeConstraint>& constraints,
                                                       const std::vector<Polynomial>& objectives) {
  // Solve a batch of ILP problems, all with the same constraints but different objectives
  Tableau t = makeStandardFormTableau(constraints);
  if (!t.convertToCanonicalForm()) {
    throw std::runtime_error("Unable to run ILPSolver::batch_solve: Feasible region empty.");
  }
  var_names_ = t.varNames();
  t.convertToCanonicalForm();

  std::map<Polynomial, ILPResult> ret;
  for (const Polynomial& obj : objectives) {
    clean();
    var_names_ = t.varNames();

    // Copy tableau for manipulation specific to the objective
    Tableau specific_t = t;

    // Set first row based on objective
    specific_t.mat()(0, 0) = 1;
    for (size_t i = 0; i < t.varNames().size(); ++i) {
      std::string var = t.varNames()[i];
      if (var.substr(var.size() - 4, 4) == "_pos") {
        specific_t.mat()(0, i + 1) = -obj[var.substr(1, var.size() - 5)];
      } else if (var.substr(var.size() - 4, 4) == "_neg") {
        specific_t.mat()(0, i + 1) = obj[var.substr(1, var.size() - 5)];
      } else {
        // Do nothing: We're on a slack variable or other artificially added variable
      }
    }

    // Since objective was reset, need to price out to make canonical
    specific_t.priceOut();
    ret.emplace(std::piecewise_construct, std::make_tuple(obj), std::make_tuple(solve(specific_t, true)));
  }
  return ret;
}

ILPResult ILPSolver::solve(const std::vector<RangeConstraint>& constraints, const Polynomial objective) {
  if (VLOG_IS_ON(2)) {
    std::ostringstream msg;
    msg << "Starting ILPSolver with constraints\n";
    for (const RangeConstraint& c : constraints) {
      msg << "  " << c << "\n";
    }
    msg << "and objective " << objective;
    IVLOG(2, msg.str());
  }
  Tableau t = makeStandardFormTableau(constraints, objective);
  return solve(t);
}

ILPResult ILPSolver::solve(Tableau& tableau, bool already_canonical) {
  clean();
  var_names_ = tableau.varNames();
  IVLOG(5, "Starting ILPSolver with tableau " << tableau.mat().toString());
  solve_step(tableau, already_canonical);
  if (!feasible_found) {
    throw std::runtime_error("No feasible solution");
  }
  return ILPResult(reportObjective(), reportSolution());
}

void ILPSolver::solve_step(Tableau& tableau, bool already_canonical) {
  // Check feasible region exists for this subproblem
  if (!tableau.makeOptimal(already_canonical)) {
    // Feasible region empty (or unbounded), no solution from this branch
    IVLOG(5, "Feasible region empty; pruning branch");
    return;
  }

  // Check the LP Relaxation objective value
  Rational obj_val = tableau.reportObjectiveValue();

  // Check if this solution is integral
  std::vector<Rational> soln = tableau.getSymbolicSolution();

  // Find the greatest fractional part
  Rational greatest_fractional = 0;
  size_t greatest_fractional_row = 0;
  for (size_t i = 1; i < tableau.mat().size1(); ++i) {
    Rational frac = tableau.mat()(i, tableau.mat().size2() - 1) - Floor(tableau.mat()(i, tableau.mat().size2() - 1));
    if (frac > greatest_fractional) {
      greatest_fractional = frac;
      greatest_fractional_row = i;
    }
  }

  if (greatest_fractional == 0) {
    // This is an integer solution!
    if (VLOG_IS_ON(3)) {
      std::ostringstream msg;
      msg << "Found new best integer solution!"
          << "  objective: " << obj_val << "\n";
      msg << "  Solution is:";
      for (size_t i = 0; i < soln.size(); ++i) {
        msg << "\n    " << tableau.varNames()[i] << ": " << soln[i];
      }
      IVLOG(5, msg.str());
      IVLOG(6, "  from tableau:" << tableau.mat().toString());
    }
    feasible_found = true;
    best_objective = obj_val;
    best_solution = soln;
  } else {
    // This is a non-integer solution; cut
    if (VLOG_IS_ON(5)) {
      std::ostringstream msg;
      msg << "Found non-integer solution;"
          << "  objective: " << obj_val << "\n";
      msg << "  Solution is:";
      for (size_t i = 0; i < soln.size(); ++i) {
        msg << "\n    " << tableau.varNames()[i] << ": " << soln[i];
      }
      IVLOG(5, msg.str());
      IVLOG(6, "  from tableau:" << tableau.mat().toString());
    }

    IVLOG(5, "Requesting Gomory cut at row " << greatest_fractional_row << " with value " << greatest_fractional);
    Tableau with_cut = addGomoryCut(tableau, greatest_fractional_row);
    IVLOG(6, "Adding Gomory cut yielded: " << with_cut.mat().toString());
    solve_step(with_cut);
  }
}

Tableau ILPSolver::addGomoryCut(const Tableau& t, size_t row) {
  IVLOG(6, "Adding Gomory cut along row " << row);
  Tableau ret(t.mat().size1() + 1, t.mat().size2() + 1, t.varNames(), &t.getOpposites());
  project(ret.mat(), range(0, t.mat().size1()), range(0, t.mat().size2() - 1)) =
      project(t.mat(), range(0, t.mat().size1()), range(0, t.mat().size2() - 1));
  project(ret.mat(), range(0, t.mat().size1()), range(t.mat().size2(), t.mat().size2() + 1)) =
      project(t.mat(), range(0, t.mat().size1()), range(t.mat().size2() - 1, t.mat().size2()));
  // Note: Assumes the uninitialized column was set to all 0s, which appears to
  // be an undocumented feature of ublas.
  for (size_t j = 0; j < t.mat().size2() - 1; ++j) {
    ret.mat()(t.mat().size1(), j) = t.mat()(row, j) - Floor(t.mat()(row, j));
  }
  ret.mat()(t.mat().size1(), t.mat().size2() - 1) = -1;
  ret.mat()(t.mat().size1(), t.mat().size2()) =
      t.mat()(row, t.mat().size2() - 1) - Floor(t.mat()(row, t.mat().size2() - 1));
  return ret;
}

Tableau ILPSolver::makeStandardFormTableau(const std::vector<RangeConstraint>& constraints,
                                           const Polynomial objective) {
  // Create the standard form linear program for minimizing objective subject to the given constraints

  std::vector<Polynomial> lp_constraints;  // The represented constraint is poly == 0
  unsigned int slack_count = 0;

  std::vector<std::string> var_names;  // Ordered list of variable names used in this Tableau

  // var_index indicates what column in the Tableau will go with the variable name
  // Note that these are indexed from 0 but have 1 as the smallest value as the first
  // column in the Tableau is for the objective and does not have a variable name
  std::map<std::string, size_t> var_index;
  for (const RangeConstraint& c : constraints) {
    Polynomial poly(c.poly);

    // Split each variable into + and - parts
    // First extract keys (i.e. var names)
    std::vector<std::string> local_vars;
    for (const auto& kvp : poly.getMap()) {
      const std::string& key = kvp.first;
      if (key != "") {  // Do nothing for constant term
        local_vars.emplace_back(key);
      }
    }

    // Replace each var with + and - parts
    for (const std::string& var : local_vars) {
      std::map<std::string, size_t>::iterator unused;
      bool added_new_var;
      poly.substitute(var, Polynomial("_" + var + "_pos") - Polynomial("_" + var + "_neg"));
      std::tie(unused, added_new_var) = var_index.emplace("_" + var + "_pos", var_index.size() + 1);
      if (added_new_var) {
        var_names.emplace_back("_" + var + "_pos");
      }
      std::tie(unused, added_new_var) = var_index.emplace("_" + var + "_neg", var_index.size() + 1);
      if (added_new_var) {
        var_names.emplace_back("_" + var + "_neg");
      }
    }

    // Make LP constraint from lower bound
    std::string slack_var = "_slack" + std::to_string(slack_count);
    lp_constraints.emplace_back(poly - Polynomial(slack_var));
    var_names.emplace_back(slack_var);
    var_index.emplace(slack_var, var_index.size() + 1);
    ++slack_count;

    // Make LP constraint from upper bound
    slack_var = "_slack" + std::to_string(slack_count);
    lp_constraints.emplace_back(poly + Polynomial(slack_var) - c.range + 1);
    var_names.emplace_back(slack_var);
    var_index.emplace(slack_var, var_index.size() + 1);
    ++slack_count;
  }

  // The tableau has a row for each lp_constraint plus a row for the objective, and a
  // column for each variable plus a column for the constant terms and a column for the objective.
  Tableau tableau(lp_constraints.size() + 1, var_index.size() + 2, var_names);

  // Put the data in the Tableau
  // First the objective:
  tableau.mat()(0, 0) = 1;
  for (const auto& kvp : objective.getMap()) {
    if (kvp.first != "") {
      // The positive and negative parts have reversed sign because the algorithm
      // needs to use -objective for the coeffs of the first row
      try {
        tableau.mat()(0, var_index.at("_" + kvp.first + "_pos")) = -kvp.second;
        tableau.mat()(0, var_index.at("_" + kvp.first + "_neg")) = kvp.second;
      } catch (const std::out_of_range& e) {
        throw std::out_of_range("Bad index given to Tableau objective: " + kvp.first);
      }
    }
  }

  // Now the constraints:
  size_t constraint_idx = 1;  // Start from 1 b/c the first row is for the object
  for (const Polynomial& poly : lp_constraints) {
    short const_sign = 1;  // NOLINT (runtime/int)
    if (poly.constant() <= 0) {
      const_sign = -1;  // Last column must be positive, so negate everything if const term is positive
    }
    for (const auto& kvp : poly.getMap()) {
      if (kvp.first == "") {
        // The negative of the constant term goes in the last column of the tableau
        tableau.mat()(constraint_idx, tableau.mat().size2() - 1) = -const_sign * -kvp.second;
      } else {
        tableau.mat()(constraint_idx, var_index.at(kvp.first)) = -const_sign * kvp.second;
      }
    }
    ++constraint_idx;
  }

  return tableau;
}

void ILPSolver::clean() {
  feasible_found = false;
  best_objective = 0;
  best_solution.clear();
  var_names_.clear();
}
}  // namespace bilp
}  // namespace lang
}  // namespace tile
}  // namespace vertexai
