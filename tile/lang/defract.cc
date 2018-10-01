#include "tile/lang/defract.h"

#include <map>
#include <set>
#include <string>

#include "base/util/logging.h"
#include "base/util/printstring.h"
#include "tile/math/matrix.h"

namespace vertexai {
namespace tile {
namespace lang {

using namespace math;  // NOLINT

static Polynomial<Rational> ConvertPoly(Polynomial<Rational> in,
                                        const std::map<std::string, Polynomial<Rational>>& polys,
                                        bool transform_constant = false) {
  Polynomial<Rational> out;
  for (const auto& kvp : in.getMap()) {
    if (kvp.first == "" && !transform_constant) {
      out += kvp.second;
    } else {
      auto it = polys.find(kvp.first);
      if (it == polys.end()) {
        throw std::runtime_error("Invalid polynomial conversion");
      }
      out += kvp.second * it->second;
    }
  }
  return out;
}

Contraction Defract(const Contraction& op, const std::vector<RangeConstraint>& cons) {
  std::vector<Polynomial<Rational>> polys;
  bool has_fract = false;
  std::set<std::string> vars;
  for (const auto& c : cons) {
    for (const auto& kvp : c.poly.getMap()) {
      if (denominator(kvp.second) != 1) {
        has_fract = true;
      }
      if (kvp.first != "") {
        vars.insert(kvp.first);
      }
    }
    polys.push_back(c.poly);
  }
  if (!has_fract) {
    IVLOG(3, "No fractions to Defract")
    return op;
  }
  std::vector<std::string> vvars(vars.begin(), vars.end());

  Matrix mat;
  Vector vec;
  bool transform_constant = false;
  std::tie(mat, vec) = FromPolynomials(polys);
  IVLOG(3, "Original Matrix: " << mat);
  IVLOG(3, "Original Vector: " << vec);
  for (auto& v_entry : vec) {
    if (denominator(v_entry) != 1) {
      // Transform constant -> constant * dummy_var with 1 <= dummy_var < 2
      transform_constant = true;
      IVLOG(3, "Non-integer offset vector " << vec << " in defractionalization. Transforming constant to variable.");
      vvars.push_back("");
      mat.resize(mat.size1() + 1, mat.size2() + 1, true);
      // NOTE: For some reason, using boost::numeric::ublas::row() breaks debug builds.
      // So instead we write the equivalent code manually.
      for (size_t c = 0; c < mat.size2(); c++) {
        mat(mat.size1() - 1, c) = 0;
      }
      vec.resize(vec.size() + 1, true);
      // NOTE: For some reason, using boost::numeric::ublas::column() breaks debug builds.
      // So instead we write the equivalent code manually.
      for (size_t r = 0; r < mat.size1(); r++) {
        mat(r, mat.size2() - 1) = vec[r];
      }
      mat(mat.size1() - 1, mat.size2() - 1) = 1;
      break;
    }
  }

  if (!HermiteNormalForm(mat)) {
    throw std::runtime_error("Unable to perform Hermite Reduction during defractionalization");
  }
  IVLOG(4, "Matrix: " << mat);

  Matrix p = prod(trans(mat), mat);
  IVLOG(4, "Product Matrix: " << p);
  if (!p.invert()) {
    throw std::runtime_error("Unable to invert Hermite Matrix");
  }
  IVLOG(4, "Inverse Product Matrix: " << p);
  Matrix d = prod(mat, p);
  IVLOG(4, "Dual Matrix: " << d);
  IVLOG(3, "Normalized Dual Matrix: " << d);
  if (!HermiteNormalForm(d)) {
    throw std::runtime_error("Unable to perform Hermite Reduction on dual during defractionalization");
  }

  std::vector<RangeConstraint> new_cons;
  std::vector<std::map<Integer, Polynomial<Rational>>> mod_polys;
  for (size_t i = 0; i < d.size2(); i++) {
    IVLOG(4, "Computing splits for " << vvars[i]);
    std::set<Integer> splits;
    // For each element, compute the 'modulos' I need
    splits.insert(1);  // Sentinel
    for (size_t j = i + 1; j < d.size2(); j++) {
      Rational r = d(i, j) / d(j, j);
      splits.insert(denominator(r));
    }
    std::vector<Integer> split_vec(splits.begin(), splits.end());
    IVLOG(4, "List of splits: " << split_vec);
    // Now, break those into constraints + polynomials
    Polynomial<Rational> cpoly;
    std::map<Integer, Polynomial<Rational>> ipoly;
    for (size_t j = 0; j < split_vec.size(); j++) {
      std::string var = vvars[i] + "_" + std::to_string(j);
      // Add a constraint for non-final cases
      if (j < split_vec.size() - 1) {
        if (split_vec[j + 1] % split_vec[j] != 0) {
          throw std::runtime_error("Unable to remove modulo operations during defractionalization");
        }
        Integer div = split_vec[j + 1] / split_vec[j];
        new_cons.emplace_back(Polynomial<Rational>(var), static_cast<int64_t>(div));
      }
      // Add an entry to the polynomial
      cpoly += split_vec[j] * Polynomial<Rational>(var);
      // Add the polynomial to a lookup table
      Integer modof = (j + 1 < split_vec.size() ? split_vec[j + 1] : 0);
      ipoly[modof] = cpoly;
    }
    IVLOG(4, "List of polys: " << ipoly);
    mod_polys.push_back(ipoly);
  }

  // Now, make replacements
  std::map<std::string, Polynomial<Rational>> replacements;
  for (size_t i = 0; i < d.size2(); i++) {
    Polynomial<Rational> poly = d(i, i) * mod_polys[i][0];
    for (size_t j = 0; j < i; j++) {
      poly += d(j, i) * mod_polys[j][denominator(d(j, i) / d(i, i))];
    }
    replacements[vvars[i]] = poly;
  }

  IVLOG(3, "Replacements = " << replacements);
  IVLOG(3, "New Constraints = " << new_cons);

  // Make new version of the op by first copying the original
  Contraction new_op(op.specs.size() - 1);
  new_op.comb_op = op.comb_op;
  new_op.agg_op = op.agg_op;
  for (size_t i = 0; i < op.specs.size(); i++) {
    for (size_t j = 0; j < op.specs[i].spec.size(); j++) {
      new_op.specs[i].spec.push_back(ConvertPoly(op.specs[i].spec[j], replacements, transform_constant));
      new_op.specs[i].id = op.specs[i].id;
    }
  }

  for (size_t i = 0; i < op.constraints.size(); i++) {
    const RangeConstraint& oc = op.constraints[i].bound;
    new_op.constraints.push_back(
        SymbolicConstraint(RangeConstraint(ConvertPoly(oc.poly, replacements, transform_constant), oc.range)));
  }
  if (transform_constant) {
    // Add constraint for the constant term if it's being transformed
    new_op.constraints.push_back(SymbolicConstraint(
        RangeConstraint(ConvertPoly(Polynomial<Rational>(1), replacements, transform_constant) - 1, 1)));
  }

  for (const RangeConstraint& c : new_cons) {
    new_op.constraints.push_back(SymbolicConstraint(c));
  }
  return new_op;
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
