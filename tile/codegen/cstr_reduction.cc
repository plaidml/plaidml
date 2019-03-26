// Copyright 2018, Intel Corporation

#include "tile/codegen/cstr_reduction.h"
#include "tile/bilp/ilp_solver.h"
#include "tile/codegen/dce.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT
using namespace math;    // NOLINT
using namespace bilp;    // NOLINT

static void EvaluatePolynomial(const Polynomial<int64_t> orig_poly, const AliasMap& alias_map, int64_t* min_value,
                               int64_t* max_value) {
  Polynomial<int64_t> poly = orig_poly.sym_eval(alias_map.idx_sources());
  int64_t min = poly.constant();
  int64_t max = poly.constant();
  const std::map<std::string, int64_t>& var_map = poly.getMap();
  const std::map<std::string, uint64_t>& idx_ranges = alias_map.idx_ranges();

  for (const auto& kvp : var_map) {
    if (kvp.first == "") {
      continue;
    }
    uint64_t range = idx_ranges.at(kvp.first);
    if (kvp.second >= 0) {
      max += kvp.second * (range - 1);
    } else {
      min += kvp.second * (range - 1);
    }
  }
  *min_value = min;
  *max_value = max + 1;
}

bool IsStencilIndex(Index* idx) {
  for (const auto& tag : idx->tags) {
    if (tag.rfind("stencil_", 0) == 0) {
      return true;
    }
  }
  return false;
}

void LightCstrReduction(const AliasMap& alias_map, Block* block) {
  IVLOG(4, "Start light-weight constraint reduction.");

  if (block->constraints.empty()) {
    return;
  }

  IVLOG(4, "Index: " << block->idxs);
  IVLOG(4, "AliasMap.idx_ranges_: " << alias_map.idx_ranges());

  // The index of constraints need to be removed
  std::vector<int> remove_list;
  int cstr_idx = 0;

  for (const auto& constraint : block->constraints) {
    // Evaluate the constraints and compute the min and max values
    int64_t min_value, max_value;
    EvaluatePolynomial(constraint, alias_map, &min_value, &max_value);
    IVLOG(4, constraint << " >= 0: min = " << min_value << ", max = " << max_value);

    // For constraint >= 0, if constraint's max value <= 0,
    // the constraint is always false. So the block is impossible, remove it.
    if (max_value <= 0) {
      block->set_tag("removed");
      IVLOG(4, "Block " << block->name << "removed.");
      break;
    }

    // For constraint >= 0, if constraint's min value >= 0,
    // the constraint is always true so that it is redundant and can be removed.
    if (min_value >= 0) {
      remove_list.emplace_back(cstr_idx);
      IVLOG(4, "Constraint " << constraint << " >= 0 is redundant. Remove it.");
    }

    // Check if we can reduce the index range
    // For a constraint: -c*x + rest >= 0
    // x must be 0 when the constraint gets max_value
    // Thus max of rest is also max_value
    // So x <= rest / c <= max_value / c
    // If max_value / c + 1 < x's current range, we can reduce the range
    for (const auto& kvp : constraint.getMap()) {
      if (kvp.first == "") {
        continue;
      }
      Index* idx = block->idx_by_name(kvp.first);
      if (IsStencilIndex(idx)) {
        continue;
      }
      int64_t coeff = -kvp.second;
      if (idx->affine.getMap().empty() && coeff > 0) {
        uint64_t range = idx->range;
        // Here max_value must be larger than 0, so we can cast it to unsigned
        if (static_cast<uint64_t>(max_value) - 1 < (range - 1) * coeff) {
          uint64_t new_range = static_cast<int>(Floor((max_value - 1) / coeff)) + 1;
          IVLOG(4, "Reduce range of " << kvp.first << " from " << range << " to " << new_range);
          idx->range = new_range;
        }
      }
    }
    ++cstr_idx;
  }

  // Remove the redundant constraints
  for (auto it = remove_list.rbegin(); it != remove_list.rend(); ++it) {
    auto cst_it = block->constraints.begin() + (*it);
    block->constraints.erase(cst_it);
  }

  IVLOG(4, "End light-weight constraint reduction.");
}

static Polynomial<Rational> PolynomialIntToRational(const Polynomial<int64_t>& src) {
  Polynomial<Rational> dest;
  const std::map<std::string, int64_t>& src_map = src.getMap();
  std::map<std::string, Rational>& dest_map = dest.mutateMap();
  for (const auto& element : src_map) {
    dest_map.emplace(element.first, Rational(element.second));
  }
  return dest;
}

static RangeConstraint ConstraintToRangeConstraint(const Polynomial<int64_t>& orig) {
  Polynomial<Rational> poly = PolynomialIntToRational(orig);
  return RangeConstraint(poly, INT64_MAX);
}

static inline size_t BlockDepth(std::string idx) {
  size_t pos = idx.find(':');
  if (pos == std::string::npos) {
    return (size_t)-1;
  }
  return std::stoi(idx.substr(1, pos - 1));
}

static inline std::string IdxPostfix(std::string idx) {
  size_t pos = idx.find(':');
  if (pos == std::string::npos) {
    return "";
  }
  return idx.substr(pos + 1);
}

void IlpCstrReduction(const AliasMap& alias_map, Block* block) {
  if (block->constraints.empty()) {
    return;
  }

  IVLOG(4, "Start constraint reduction using ILP.");

  std::vector<RangeConstraint> constraints;
  std::set<std::string> used_idx;

  // Use all original constraints for new range constraints.
  for (const auto& constraint : block->constraints) {
    Polynomial<int64_t> new_cstr = constraint.sym_eval(alias_map.idx_sources());
    for (const auto& idx : new_cstr.getMap())
      if (idx.first != "") {
        used_idx.insert(idx.first);
      }
    IVLOG(4, "Constraint: " << new_cstr << " >= 0");
    constraints.emplace_back(ConstraintToRangeConstraint(new_cstr));
  }

  std::vector<Polynomial<Rational>> objectives;
  size_t current_depth = alias_map.depth();

  // Use all index ranges which are used in the constraints.
  // AliasMap.idx_range() contains only index x:r,
  // generate the range constraints: 0 <= x < r, and ojectives: x and -x for only this level
  for (const auto& idx : alias_map.idx_ranges()) {
    if (used_idx.find(idx.first) != used_idx.end()) {
      constraints.emplace_back(RangeConstraint(Polynomial<Rational>(idx.first), idx.second));
      IVLOG(4, "Constraint: 0 <= " << idx.first << " < " << idx.second);
      if (BlockDepth(idx.first) == current_depth) {
        objectives.emplace_back(Polynomial<Rational>(idx.first, -1));
        IVLOG(4, "Objective: min: -" << idx.first);
      }
    }
  }

  ILPSolver solver;
  solver.set_throw_infeasible(false);

  std::map<Polynomial<Rational>, ILPResult> results = solver.batch_solve(constraints, objectives);

  if (results.size() == 0) {
    // The block is infeasible
    block->set_tag("removed");
    IVLOG(4, "Remove redundant block " << block->name);
    return;
  } else {
    // Check if the idx range can be reduced
    for (const auto& objective : objectives) {
      const std::string& idx_name = objective.getMap().begin()->first;
      Index* idx = block->idx_by_name(IdxPostfix(idx_name));
      if (IsStencilIndex(idx)) {
        continue;
      }
      uint64_t new_range = static_cast<uint64_t>(Floor(-results[objective].obj_val)) + 1;
      if (new_range < idx->range) {
        IVLOG(4, "Reduce range of " << idx_name << " from " << idx->range << " to " << new_range);
        idx->range = new_range;
      }
    }
  }

  // constraint iterator in constraints
  auto cstr_iter = constraints.begin();
  // constraint index in block->constraints
  int cstr_idx_blk = 0;
  int total_cstr = block->constraints.size();
  std::vector<int> remove_list;

  while (cstr_idx_blk < total_cstr) {
    RangeConstraint tmp_cst = *cstr_iter;
    auto next_cstr_iter = constraints.erase(cstr_iter);
    Polynomial<Rational>& objective = tmp_cst.poly;

    ILPResult result = solver.solve(constraints, objective);
    // objective value doesn't containt the constant, is it a bug?
    Rational obj_val = result.obj_val + objective.constant();

    // If the objective >= 0, the constraint is redundant,
    // do not restore it in constraint list.
    // Otherwise, restore the constraints into the list
    if (obj_val >= 0) {
      remove_list.emplace_back(cstr_idx_blk);
      cstr_iter = next_cstr_iter;
    } else {
      constraints.insert(next_cstr_iter, tmp_cst);
      cstr_iter = next_cstr_iter + 1;
    }
    ++cstr_idx_blk;
  }

  // Traverse reversely. Otherwise, the index may not be correct after deletion
  for (int i = remove_list.size() - 1; i >= 0; --i) {
    auto it = block->constraints.begin() + remove_list[i];
    IVLOG(4, "Remove redundant constraint: " << *it);
    block->constraints.erase(it);
  }
  IVLOG(4, "End constraint reduction using ILP.");
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
