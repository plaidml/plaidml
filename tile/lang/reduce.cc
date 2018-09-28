
#include "tile/lang/reduce.h"

#include <set>
#include <string>

#include "base/util/logging.h"
#include "base/util/printstring.h"
#include "tile/lang/compile.h"
#include "tile/lang/ops.h"
#include "tile/math/basis.h"
#include "tile/math/matrix.h"

namespace vertexai {
namespace tile {
namespace lang {

static Polynomial<Rational> ConvertVariables(Polynomial<Rational> in, std::vector<std::string> vars,
                                             std::vector<Polynomial<Rational>> polys) {
  Polynomial<Rational> out;
  for (size_t i = 0; i < vars.size(); i++) {
    out += in[vars[i]] * polys[i];
  }
  out += in.constant();
  return out;
}

Contraction ReduceOutputPolynomials(const Contraction& op, const std::vector<RangeConstraint>& order) {
  // First, we find all of our index variables
  // And also, keep track of variables used in output
  std::set<std::string> output_variables;
  std::set<std::string> index_variables;
  std::tie(index_variables, output_variables) = op.getIndexAndOutputVars();

  // Now we construct a set of 'rewrite' variables
  // for each linearly independent output Polynomial<Rational>
  BasisBuilder basis;
  for (const Polynomial<Rational>& p : op.specs[0].spec) {
    // Maybe add it to the equation list
    basis.addEquation(p);
  }
  // Next, fill in from contraints until we have as many as variables
  // or we run out of options
  for (const auto& con : order) {
    if (basis.dimensions() == index_variables.size()) {
      break;
    }
    basis.addEquation(con.poly);
  }
  const std::vector<Polynomial<Rational>>& forms = basis.basis();
  if (forms.size() < index_variables.size()) {
    throw std::runtime_error("Underspecified set of equations in index variables");
  }

  // Print out equations
  if (VLOG_IS_ON(3)) {
    VLOG(3) << "In reduce, intial equations are";
    for (size_t i = 0; i < forms.size(); i++) {
      VLOG(3) << "  v" << i << " == " << forms[i].toString();
    }
  }

  // Convert to a matrix form + invert
  size_t size = forms.size();
  Matrix m;
  std::tie(m, std::ignore) = FromPolynomials(forms);
  if (!m.invert()) {
    throw std::runtime_error("Attempt to solve indexing equations failed due to singular matrix");
  }

  // Now convert back to equations
  std::vector<Polynomial<Rational>> inverses;
  for (size_t i = 0; i < size; i++) {
    Polynomial<Rational> p;
    for (size_t j = 0; j < size; j++) {
      p += m(i, j) * Polynomial<Rational>(std::string("v") + std::to_string(j));
    }
    inverses.push_back(p);
  }

  // Vectorize variables
  std::vector<std::string> vec_idx(index_variables.begin(), index_variables.end());

  // Print new equations
  if (VLOG_IS_ON(3)) {
    VLOG(3) << "In reduce, reduced equations are:";
    for (size_t i = 0; i < size; i++) {
      VLOG(3) << "  " << vec_idx[i] << " == " << to_string(inverses[i]);
    }
  }

  // Make new version of the op by first copying the original
  Contraction new_op(op.specs.size() - 1);
  new_op.comb_op = op.comb_op;
  new_op.agg_op = op.agg_op;
  for (size_t i = 0; i < op.specs.size(); i++) {
    for (size_t j = 0; j < op.specs[i].spec.size(); j++) {
      new_op.specs[i].spec.push_back(ConvertVariables(op.specs[i].spec[j], vec_idx, inverses));
      new_op.specs[i].id = op.specs[i].id;
    }
  }

  for (size_t i = 0; i < op.constraints.size(); i++) {
    const RangeConstraint& oc = op.constraints[i].bound;
    new_op.constraints.push_back(
        SymbolicConstraint(RangeConstraint(ConvertVariables(oc.poly, vec_idx, inverses), oc.range)));
  }
  return new_op;
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
