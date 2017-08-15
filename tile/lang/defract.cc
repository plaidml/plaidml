
#include "tile/lang/defract.h"

#include <map>
#include <set>
#include <string>

#include "base/util/logging.h"
#include "base/util/printstring.h"
#include "tile/lang/matrix.h"

namespace vertexai {
namespace tile {
namespace lang {

static Polynomial ConvertPoly(Polynomial in, const std::map<std::string, Polynomial>& polys) {
  Polynomial out;
  for (const auto& kvp : in.getMap()) {
    if (kvp.first == "") {
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
  std::vector<Polynomial> polys;
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

  Matrix m;
  Vector v;
  std::tie(m, v) = FromPolynomials(polys);
  IVLOG(3, "Original Matrix: " << m);
  IVLOG(3, "Original Vector: " << v);

  if (!HermiteNormalForm(m, v)) {
    throw std::runtime_error("Unable to peform Hermite Reduction during defractionalization");
  }
  IVLOG(4, "Matrix: " << m);
  IVLOG(4, "Vector: " << v);

  Matrix p = prod(trans(m), m);
  IVLOG(4, "Product Matrix: " << p);
  if (!p.invert()) {
    throw std::runtime_error("Unable to invert Hermite Matrix");
  }
  IVLOG(4, "Inverse Product Matrix: " << p);
  Matrix d = prod(m, p);
  IVLOG(4, "Dual Matrix: " << d);
  IVLOG(3, "Normalized Dual Matrix: " << d);
  if (!HermiteNormalForm(d, v)) {
    throw std::runtime_error("Unable to peform Hermite Reduction on dual during defractionalization");
  }

  std::vector<RangeConstraint> new_cons;
  std::vector<std::map<Integer, Polynomial>> mod_polys;
  for (size_t i = 0; i < d.size2(); i++) {
    IVLOG(4, "Computing splits for " << vvars[i]);
    std::set<Integer> splits;
    // For each element, compute the 'modulos' I need
    splits.insert(1);  // Sentinal
    for (size_t j = i + 1; j < d.size2(); j++) {
      Rational r = d(i, j) / d(j, j);
      splits.insert(denominator(r));
    }
    std::vector<Integer> split_vec(splits.begin(), splits.end());
    IVLOG(4, "List of splits: " << split_vec);
    // Now, break those into constraints + polynomials
    Polynomial cpoly;
    std::map<Integer, Polynomial> ipoly;
    for (size_t j = 0; j < split_vec.size(); j++) {
      std::string var = vvars[i] + "_" + std::to_string(j);
      // Add a constraint for non-final cases
      if (j < split_vec.size() - 1) {
        if (split_vec[j + 1] % split_vec[j] != 0) {
          throw std::runtime_error("Unable to remove modulo operations during defractionalization");
        }
        Integer div = split_vec[j + 1] / split_vec[j];
        new_cons.emplace_back(Polynomial(var), static_cast<int64_t>(div));
      }
      // Add an entry to the polynomial
      cpoly += split_vec[j] * Polynomial(var);
      // Add the polynomial to a lookup table
      Integer modof = (j + 1 < split_vec.size() ? split_vec[j + 1] : 0);
      ipoly[modof] = cpoly;
    }
    IVLOG(4, "List of polys: " << ipoly);
    mod_polys.push_back(ipoly);
  }

  // Now, make replacements
  std::map<std::string, Polynomial> replacements;
  for (size_t i = 0; i < d.size2(); i++) {
    Polynomial poly = d(i, i) * mod_polys[i][0];
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
      new_op.specs[i].spec.push_back(ConvertPoly(op.specs[i].spec[j], replacements));
      new_op.specs[i].id = op.specs[i].id;
    }
  }

  for (size_t i = 0; i < op.constraints.size(); i++) {
    const RangeConstraint& oc = op.constraints[i].bound;
    new_op.constraints.push_back(SymbolicConstraint(RangeConstraint(ConvertPoly(oc.poly, replacements), oc.range)));
  }

  for (const RangeConstraint& c : new_cons) {
    new_op.constraints.push_back(SymbolicConstraint(c));
  }
  return new_op;
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
