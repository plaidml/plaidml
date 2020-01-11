#include "pmlc/util/bilp/tableau.h"

#include <set>

namespace vertexai {
namespace tile {
namespace bilp {

using namespace math;  // NOLINT

Tableau::Tableau(const Matrix& m, const std::vector<std::string>& var_names, const std::vector<size_t>* opposites)
    : matrix_(m), var_names_(var_names), opposites_(var_names.size(), 0) {
  if (opposites) {
    opposites_ = *opposites;
  } else {
    buildOppositesFromNames();
  }
}

Tableau::Tableau(Matrix::size_type size1, Matrix::size_type size2, const std::vector<std::string>& var_names,
                 const std::vector<size_t>* opposites)
    : matrix_(size1, size2), var_names_(var_names), opposites_(var_names.size(), 0) {
  if (opposites) {
    opposites_ = *opposites;
  } else {
    buildOppositesFromNames();
  }
}

RowToColLookup Tableau::basicVars() const { return basic_vars_; }

std::vector<std::string> Tableau::varNames() const { return var_names_; }

void Tableau::buildOppositesFromNames() {
  for (size_t i = 0; i < var_names_.size(); ++i) {
    if (var_names_[i].substr(var_names_[i].length() - 4, 4) == "_pos" && var_names_[i][0] == '_') {
      for (size_t j = i + 1; j < var_names_.size(); ++j) {
        if (var_names_[j] == var_names_[i].substr(0, var_names_[i].length() - 4) + "_neg") {
          opposites_[i] = j;
          opposites_[j] = i;
          break;
        }
      }
    } else if (var_names_[i].substr(var_names_[i].length() - 4, 4) == "_neg" && var_names_[i][0] == '_') {
      for (size_t j = i + 1; j < var_names_.size(); ++j) {
        if (var_names_[j] == var_names_[i].substr(0, var_names_[i].length() - 4) + "_pos") {
          opposites_[i] = j;
          opposites_[j] = i;
          break;
        }
      }
    }
  }
}

bool Tableau::convertToCanonicalForm() {
  Tableau phase1(mat().size1(), mat().size2() + mat().size1(), var_names_, &opposites_);
  phase1.mat()(0, 0) = 1;

  // Put the size1() - 1 artificial variables in columns 1 through size1() - 1
  for (size_t j = 1; j < mat().size1(); ++j) {
    phase1.mat()(0, j) = -1;
    phase1.mat()(j, j) = 1;
  }

  // Copy in original tableau (minus objective row)
  project(phase1.mat(), range(1, phase1.mat().size1()), range(mat().size1(), phase1.mat().size2())) =
      project(mat(), range(1, mat().size1()), range(0, mat().size2()));

  phase1.selectBasicVars();
  phase1.priceOut();
  if (!phase1.makeOptimal(true)) {
    throw std::runtime_error(
        "Unable to convert LP tableau to canonical form, likely due to unbounded feasible region.");
  }
  Rational optimum = phase1.mat()(0, phase1.mat().size2() - 1);
  if (optimum == 0) {
    // Ensure no artificial variables remain basic
    bool has_artificial_basic = true;  // Test at least once
    while (has_artificial_basic) {
      has_artificial_basic = false;
      for (const auto& kvp : phase1.basicVars()) {
        if (kvp.second < mat().size1()) {
          // Drive out
          for (size_t j = mat().size1(); j < phase1.mat().size2() - 1; ++j) {
            if (phase1.mat()(kvp.first, j) != 0) {
              // Can drive out to here
              // Since this is already optimized and the artificial basic is 0,
              // we can pivot here without messing anything up without further checks
              phase1.basic_vars_[kvp.first] = j;
              phase1.mat().makePivotAt(kvp.first, j);
              break;
            }
          }

          has_artificial_basic = true;
          break;
        }
      }
    }

    // Set current tableau to found canonical form
    project(mat(), range(1, mat().size1()), range(0, mat().size2())) =
        project(phase1.mat(), range(1, phase1.mat().size1()), range(mat().size1(), phase1.mat().size2()));

    // Set basic variables
    selectBasicVars();
    priceOut();
    return true;
  } else if (optimum > 0) {
    // Feasible region is empty and there is no solution to original problem
    return false;
  } else if (optimum < 0) {
    throw std::runtime_error("Internal error: Optimized LP artificial variables below 0.");
  } else {
    throw std::runtime_error("Internal error: LP optimum not trichotomous.");
  }
}

bool Tableau::makeOptimal(bool already_canonical) {
  // Convert to an equivalent tableau giving optimal real solution
  if (!already_canonical) {
    if (!convertToCanonicalForm()) {
      // Feasible region is empty
      return false;
    }
  }

  // Select pivot col
  Rational max_obj_coeff = 0;  // If the max is <= 0, we're done, so feel free to start there
  size_t max_obj_col = 0;      // 0 is not a valid column, so ok to start at this

  std::set<size_t> nonbasic_cols;
  for (size_t j = 1; j < mat().size2() - 1; ++j) {
    nonbasic_cols.insert(nonbasic_cols.end(), j);
  }
  for (const auto& kvp : basic_vars_) {
    nonbasic_cols.erase(kvp.second);
  }
  for (const size_t& j : nonbasic_cols) {
    // Skipping the first and last columns which aren't variables
    Rational curr_coeff = mat()(0, j);
    if (max_obj_coeff < curr_coeff) {
      max_obj_coeff = curr_coeff;
      max_obj_col = j;
    }
  }
  if (max_obj_coeff == 0) {
    // Tableau now optimized
    return true;
  }

  // Select pivot row
  Rational min_pivot_ratio = 0;  // Will separately initialize when first used
  size_t min_pivot_row = 0;      // 0 is not a valid row, so ok to start at this
  for (size_t i = 1; i < mat().size1(); ++i) {
    // Skipping the first row which isn't a constraint
    if (mat()(i, max_obj_col) > 0) {
      if (min_pivot_row == 0) {
        // This is the first row with positive coeff, so it's current min
        min_pivot_row = i;
        min_pivot_ratio = mat()(i, mat().size2() - 1) / mat()(i, max_obj_col);
      } else {
        Rational ratio = mat()(i, mat().size2() - 1) / mat()(i, max_obj_col);
        if (ratio < min_pivot_ratio) {
          min_pivot_row = i;
          min_pivot_ratio = ratio;
        }
      }
    }
  }
  if (min_pivot_row == 0) {
    // Feasible region unbounded; tableau has no optimum
    return false;
  }

  basic_vars_[min_pivot_row] = max_obj_col;
  mat().makePivotAt(min_pivot_row, max_obj_col);
  // iterate
  return makeOptimal(true);
}

void Tableau::selectBasicVars() {
  // Makes basic_vars_ a map pointing from each row (other than 1st) to the basic var column for it
  basic_vars_ = RowToColLookup();  // Start from scratch
  if (mat().size1() - 1 == 0) {
    // No basic variables to find!
    return;
  }
  for (size_t j = 1; j < mat().size2() - 1; ++j) {
    // Skipping the first and last columns which can't be basic vars
    // Otherwise, search each column; if it contains all 0s and one 1, it's basic
    size_t row_with_1 = 0;  // A 1 in the objective won't count, so we can also use it for
                            // columns where no 1 is ever found.
    bool is_basic = true;
    for (size_t i = 1; i < mat().size1(); ++i) {
      // Ok if objective is nonzero, but will need to price out later, so start at 1
      Rational entry = mat()(i, j);
      if (entry == 1) {
        if (row_with_1 == 0) {
          row_with_1 = i;
          continue;
        } else {
          is_basic = false;
          break;
        }
      } else if (entry == 0) {
        continue;
      } else {
        is_basic = false;
        break;
      }
    }
    if (row_with_1 == 0) {
      is_basic = false;
    }
    if (is_basic) {
      // Select first basic variable when multiple choices exist
      if (!basic_vars_.count(row_with_1)) {
        basic_vars_[row_with_1] = j;
        // Price out any nonzero objective
        mat().addRowMultToRow(0, row_with_1, -mat()(0, j) / mat()(row_with_1, j));
      }
      if (basic_vars_.size() == mat().size1() - 1) {
        // We've found them all
        return;
      }
    }
  }
  // We didn't find enough basic variables.
  throw std::runtime_error("Tableau::selectBasicVars called on a Tableau " + mat().toString() +
                           " with too few basic vars (found " + std::to_string(basic_vars_.size()) + ", need " +
                           std::to_string(mat().size1() - 1) + ").");
}

void Tableau::priceOut() {
  for (const auto& kvp : basic_vars_) {
    mat().addRowMultToRow(0, kvp.first, -mat()(0, kvp.second) / mat()(kvp.first, kvp.second));
  }
}

std::vector<Rational> Tableau::getSymbolicSolution() const {
  std::vector<Rational> soln(var_names_.size(), 0);
  for (const auto& kvp : basic_vars_) {
    if (kvp.second <= var_names_.size()) {
      soln[kvp.second - 1] = mat()(kvp.first, mat().size2() - 1);
    }
  }
  return soln;
}

Rational Tableau::reportObjectiveValue() const { return mat()(0, mat().size2() - 1); }

}  // namespace bilp
}  // namespace tile
}  // namespace vertexai
