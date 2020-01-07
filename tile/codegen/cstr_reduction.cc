// Copyright 2018, Intel Corporation

#include "tile/codegen/cstr_reduction.h"

#include <algorithm>
#include <cassert>
#include <string>

#include "base/util/any_factory_map.h"
#include "tile/bilp/ilp_solver.h"
#include "tile/codegen/dce.h"
#include "tile/math/util.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT
using namespace math;    // NOLINT
using namespace bilp;    // NOLINT

static void EvaluatePolynomial(const Polynomial<int64_t> orig_poly, const AliasMap& alias_map, int64_t* min_value,
                               int64_t* lim_value) {
  Polynomial<int64_t> poly = orig_poly.sym_eval(alias_map.idx_sources());
  int64_t min = 0;
  int64_t max = 0;

  for (const auto& kvp : poly.getMap()) {
    if (kvp.first == "") {
      min += kvp.second;
      max += kvp.second;
      continue;
    }
    uint64_t range = alias_map.idx_ranges().at(kvp.first);
    if (kvp.second >= 0) {
      max += kvp.second * (range - 1);
    } else {
      min += kvp.second * (range - 1);
    }
  }
  *min_value = min;
  *lim_value = max + 1;
}

void LightCstrReduction(const AliasMap& alias_map, Block* block, const proto::LightConstraintReductionPass& options) {
  if (block->constraints.empty()) {
    return;
  }
  IVLOG(4, "Start light-weight constraint reduction:\n" << *block);
  IVLOG(4, "Index: " << block->idxs);
  IVLOG(4, "AliasMap.idx_ranges_: " << alias_map.idx_ranges());

  auto skip_idxs = stripe::FromProto(options.skip_idxs());

  for (const auto& constraint : block->constraints) {
    // Evaluate the constraints and compute the min and lim values
    int64_t min_value, lim_value;
    EvaluatePolynomial(constraint, alias_map, &min_value, &lim_value);
    IVLOG(4, "con=" << constraint << " >= 0: min=" << min_value << " lim=" << lim_value);

    // For constraint >= 0, if constraint's lim value <= 0,
    // the constraint is always false. So the block is impossible, remove it.
    if (lim_value <= 0) {
      block->set_tag("removed");
      IVLOG(4, "Block " << block->name << "removed.");
      break;
    }

    IVLOG(4, "Squeezing constraint indices");
    const auto& constraint_map = constraint.getMap();
    for (auto kvp_it = constraint_map.begin(); kvp_it != constraint_map.end();) {
      const auto& kvp = *kvp_it++;
      if (kvp.first == "") {
        continue;
      }
      Index* idx = block->idx_by_name(kvp.first);
      if (idx->has_any_tags(skip_idxs)) {
        IVLOG(4, "  Skipping index=" << *idx << " (skip_idxs=" << skip_idxs << ")");
        continue;
      }
      if (!idx->affine.getMap().empty()) {
        IVLOG(4, "  Skipping non-constant index=" << *idx);
      }
      int64_t coeff = kvp.second;
      IVLOG(4, "  Considering index=" << *idx << " with coeff=" << coeff);

      // Check if we can reduce the index range by considering the constraint's lim value.
      //
      // Given an idx:
      //     coeff*idx + (rest of the constraint) >= 0
      // And assuming:
      //     coeff < 0
      // .. i.e. when the constraint is == lim_value-1, idx == 0
      //    i.e. increasing idx only makes the overall constraint decrease.
      // Then: coeff*idx >= -rest
      //       -coeff*idx <= rest
      //       idx <= rest / -coeff  (rounding division downwards -- N.B. -coeff is positive)
      // And since the maximum value (rest) might take on is lim_value-1,
      //       idx <= lim_value-1 / -coeff
      // So:
      //       range'(idx) = min(range(idx), (lim_value-1 / -coeff) + 1)
      if (coeff < 0) {
        uint64_t range = idx->range;
        // N.B. 0 < lim_value, or the block would've already been marked for removal.
        uint64_t new_range = std::min(range, (static_cast<uint64_t>(lim_value - 1) / -coeff) + 1);
        if (range != new_range) {
          IVLOG(4, "Reduce range of " << kvp.first << " from " << range << " to " << new_range);
          idx->range = new_range;
          min_value += (range - new_range) * -coeff;
          IVLOG(4, "  New min=" << min_value);
        }
      }

      // Check if we can reduce the index range by considering the constraint's min value.
      //
      // Given an idx:
      //     coeff*idx + (rest of the constraint) >= 0
      // And assuming:
      //     0 < coeff
      // .. i.e. when the constraint is == min_value, idx == 0
      //    i.e. increasing idx only makes the overall constraint increase.
      // Then:
      //     coeff*idx >= -rest
      //     idx >= -rest / coeff  (rounding upwards -- N.B. coeff is positive)
      //
      // To establish a lower bound on idx (since we're attempting to trim off an infeasible range of
      // [0...something)), we need the minimum value that -(rest) might take on:
      //     min(-rest)
      //     = -(max(rest))
      //     = -(max(contraction) - max(idx))
      //     = max(idx) - max(contraction)
      //     = ((coeff * range(idx)) - 1) - max(contraction)
      //     = ((coeff * range(idx)) - 1) - (lim - 1)
      //     = (coeff * range(idx)) - lim
      //
      // So putting it together, the feasible region looks like:
      //     idx >= ((coeff * range(idx)) - lim_value) / coeff  (rounding upwards)
      //     idx >= range(idx) - (lim_value / coeff)            (rounding upwards)
      //
      // ... and the range [0, range(idx) - (lim_value / coeff)) is infeasible.
      //
      // Define:
      //     shift = range(idx) - RoundUp(lim_value, coeff)
      //
      // If shift > 0, we can rewrite all uses of the index, eliding the [0..shift) range that is being
      // skipped over by this constraint:
      //     idx' = idx + shift
      //     range'(idx) = range(idx) - shift
      //
      // Note that rewriting the use of the index within the current constraint will also affect the current
      // min_value and lim_value, by adding the shift to each.
      //
      // Also, note that reducing the range of the index may cause other constraints (potentially
      // already-processed contraints) to become redundant -- which is why we perform the redundant constraint
      // check after squeezing indicies across all constraints.
      //
      // TODO: Consider removing the index entirely when when shift == range(idx), since the index will only
      //       ever take on the value of zero.
      if (0 < coeff) {
        std::int64_t shift = static_cast<int64_t>(idx->range) - RoundUp(lim_value, coeff);
        if (0 < shift) {
          IVLOG(4, "Shifting " << kvp.first << " by " << shift);
          idx->range -= shift;
          std::string idx_name = kvp.first;
          Affine new_idx{idx_name};
          new_idx += shift;
          for (auto& con : block->constraints) {
            con.substitute(idx_name, new_idx);
          }
          for (auto& dev : block->location.devs) {
            for (auto& unit : dev.units) {
              unit.substitute(idx_name, new_idx);
            }
          }
          for (auto& ref : block->refs) {
            for (auto& access : ref.mut().access) {
              access.substitute(idx_name, new_idx);
            }
            if (ref.cache_unit) {
              ref.mut().cache_unit->substitute(idx_name, new_idx);
            }
          }
          for (auto& stmt : block->stmts) {
            if (auto sub = Block::Downcast(stmt)) {
              for (auto& sidx : block->idxs) {
                sidx.affine.substitute(idx_name, new_idx);
              }
            } else if (auto li = LoadIndex::Downcast(stmt)) {
              li->from.substitute(idx_name, new_idx);
            }
          }
          min_value += (shift * coeff);
          lim_value += (shift * coeff);
          IVLOG(4, "  New idx=" << *idx);
          IVLOG(4, "  New con=" << constraint << " >= 0: min=" << min_value << " lim=" << lim_value);
        }
      }
    }
  }

  // Remove the redundant constraints.  Note that we only do this after squeezing indices across all
  // constraints, so that the updated ranges may be applied.
  //
  // Also: note that since we're removing elements from a vector, we walk the constraints in reverse order.
  // We use a forward iterator for this to make the iterator updates upon erasure more explicit.

  // TODO: It would be better to update the original alias_map's index ranges, so that inner blocks would
  // observe the constrained index space.
  AliasMap adjusted_alias_map{*alias_map.parent_alias_map(), block};
  for (auto c_it = block->constraints.end(); c_it != block->constraints.begin();) {
    --c_it;
    int64_t min_value, lim_value;
    EvaluatePolynomial(*c_it, adjusted_alias_map, &min_value, &lim_value);
    if (0 <= min_value) {
      assert(0 < lim_value);
      // The constraint is always true; it is redundant and can be removed.
      IVLOG(4, "Constraint " << *c_it << " >= 0 is redundant; removing it.");
      c_it = block->constraints.erase(c_it);
    }
  }

  IVLOG(4, "End light-weight constraint reduction:\n" << *block);
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

void IlpCstrReduction(const AliasMap& alias_map, Block* block, const proto::IlpConstraintReductionPass& options) {
  if (block->constraints.empty()) {
    return;
  }

  IVLOG(4, "Start constraint reduction using ILP.");

  std::vector<RangeConstraint> constraints;
  std::set<std::string> used_idx;
  auto skip_idxs = stripe::FromProto(options.skip_idxs());

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
      if (idx->has_tags(skip_idxs)) {
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

void LightCstrReductionPass::Apply(CompilerState* state) const {
  auto reqs = stripe::FromProto(options_.reqs());
  RunOnBlocks(state->entry(), reqs,
              [this](const AliasMap& alias_map, stripe::Block* block) {  //
                LightCstrReduction(alias_map, block, options_);
              },
              true);
}

void IlpCstrReductionPass::Apply(CompilerState* state) const {
  auto reqs = stripe::FromProto(options_.reqs());
  RunOnBlocks(state->entry(), reqs,
              [this](const AliasMap& alias_map, stripe::Block* block) {  //
                IlpCstrReduction(alias_map, block, options_);
              },
              true);
}

namespace {
[[gnu::unused]] char reg = []() -> char {
  CompilePassFactory<LightCstrReductionPass, proto::LightConstraintReductionPass>::Register();
  CompilePassFactory<IlpCstrReductionPass, proto::IlpConstraintReductionPass>::Register();
  return 0;
}();
}  // namespace
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
