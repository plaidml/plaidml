// Copyright 2019, Intel Corporation

#include "pmlc/dialect/tile/contraction.h"

#include <limits>

#include "llvm/Support/FormatVariadic.h"

#include "base/util/logging.h"
#include "tile/bilp/ilp_solver.h"
#include "tile/math/basis.h"

namespace bilp = vertexai::tile::bilp;

using vertexai::tile::math::Integer;
using vertexai::tile::math::RangeConstraint;
using vertexai::tile::math::Rational;

namespace pmlc {
namespace dialect {
namespace tile {

static bool IsImplied(const math::SimpleConstraint& constraint, const IndexBounds& bounds) {
  auto worst = constraint.poly.constant();
  for (const auto& kvp : constraint.poly.getMap()) {
    if (kvp.first.empty()) {
      continue;
    }
    if (kvp.second < 0) {
      worst += kvp.second * bounds.find(kvp.first)->second.min;
    } else {
      worst += kvp.second * bounds.find(kvp.first)->second.max;
    }
  }
  return (worst <= constraint.rhs);
}

static Rational UnifiedOffset(const Rational& c1, const Rational& c2, const Integer& n1, const Integer& n2) {
  std::set<Rational> offsets;
  if (n1 > std::numeric_limits<std::size_t>::max() || n2 > std::numeric_limits<std::size_t>::max()) {
    throw std::out_of_range("Cannot unify offset when relative quotient exceeds size_t.");
  }
  for (size_t i = 0; i < math::Abs(n1); ++i) {
    offsets.insert(std::end(offsets), math::FracPart((c1 + i) / n1));
  }
  for (size_t j = 0; j < math::Abs(n2); ++j) {
    Rational offset = math::FracPart((c2 + j) / n2);
    if (offsets.count(offset)) {
      return offset;
    }
  }
  IVLOG(1, "Failed to compute UnifiedOffset(" << c1 << ", " << c2 << ", " << n1 << ", " << n2 << ").");
  throw std::runtime_error("Merging constraints with empty intersection.");
}

static RangeConstraint IntersectParallelConstraintPair(  //
    const RangeConstraint& constraint1,                  //
    const RangeConstraint& constraint2) {
  // Combines two parallel constraints into one. See merge-parallel.tex in
  // /tile/lang for more details.
  IVLOG(5, "Merging the parallel constraints " << constraint1 << ", " << constraint2);
  Rational ratio = constraint1.poly.tryDivide(constraint2.poly, true);
  if (ratio == 0) {
    throw std::invalid_argument("Parameters of IntersectParallelConstraintPair must be parallel");
  }
  Integer n1 = numerator(ratio);
  Integer n2 = denominator(ratio);
  Rational c1 = constraint1.poly.constant();
  Rational c2 = constraint2.poly.constant();
  // d is the fractional part of the offset of merged constraint polynomial
  Rational d = UnifiedOffset(c1, c2, n1, n2);
  // Range unification requires solving the following equations for q:
  //    n1*q + c1 = 0           n2*q + c2 = 0
  //    n1*q + c1 = r1 - 1      n2*q + c2 = r2 - 1
  Rational q1_low = math::Min(-c1 / n1, (constraint1.range - 1 - c1) / n1);
  Rational q1_hi = math::Max(-c1 / n1, (constraint1.range - 1 - c1) / n1);
  Rational q2_low = math::Min(-c2 / n2, (constraint2.range - 1 - c2) / n2);
  Rational q2_hi = math::Max(-c2 / n2, (constraint2.range - 1 - c2) / n2);
  Integer lower_bound = math::Max(math::Ceil(q1_low + d), math::Ceil(q2_low + d));
  Integer upper_bound = math::Min(math::Floor(q1_hi + d), math::Floor(q2_hi + d));
  Rational merged_offset = -lower_bound + d;
  Integer range = upper_bound - lower_bound + 1;
  if (range <= 0) {
    throw std::runtime_error("Merging constraints with empty intersection: " + to_string(constraint1) + ", " +
                             to_string(constraint2));
  }
  if (range > INT64_MAX) {
    throw std::out_of_range("Bound range in IntersectParallelConstraintPair overflows int64.");
  }
  int64_t r = (int64_t)range;
  IndexPoly p(constraint1.poly / n1);
  p.setConstant(merged_offset);
  return RangeConstraint(p, r);
}

void Constraints::AddTensor(const IndexAccess& access, stripe::TensorType tensorType) {
  auto shape = tensorType.getShape();
  if (access.size() != shape.size()) {
    throw std::runtime_error(str(boost::format("Indexes != dimensions: %1% != %2%") % access.size() % shape.size()));
  }
  for (size_t i = 0; i < access.size(); i++) {
    constraints.emplace_back(access[i], shape[i].size);
  }
}

void Constraints::MergeParallelConstraints() {
  auto i = constraints.begin();
  while (i != constraints.end()) {
    // Remove trivially true constraints, fail for trivially false constraints
    // By trivially true/false, I mean constraints of the form 0 <= c < r,
    // where c is a number rather than a polynomial.
    if (i->poly.GetNonzeroIndex().empty()) {
      if (0 <= i->poly.constant() && i->poly.constant() < i->range) {
        // Constraint is trivially true; remove and continue
        if (i == constraints.begin()) {
          constraints.erase(i);
          i = constraints.begin();
          continue;
        } else {  // i-- exists
          i--;
          constraints.erase(i + 1);
          i++;  // This is the same value of i we would have checked next if not deleting
          continue;
        }
      } else {
        // Constraint is trivially false; fail indicating no solutions
        std::string ErrorMessage = "Error: Always false constraint given to MergeParallelConstraints.";
        ErrorMessage += "\nConstraint poly: " + i->poly.toString();
        ErrorMessage += "\nConstraint range: " + std::to_string(i->range);
        throw std::invalid_argument(ErrorMessage);
      }
    }
    for (auto j = i + 1; j != constraints.end(); ++j) {
      if (i->IsParallel(*j)) {
        (*i) = IntersectParallelConstraintPair(*i, *j);

        // Decrement j so it stays valid after erase, then erase where j
        // used to be. Incrementing this j at the end of the loop body
        // gives the same element as if we hadn't deleted the original j
        //
        // Slightly inefficient to repeatedly erase; could instead
        // create a list of iterators pointing at things to erase, then
        // erase them all at the end. But there's some added complexity
        // to that and I don't think it gains us much speed, so I'm
        // not doing that (for now? [TODO perf (minor)])
        j--;  // must exist since i exists
        constraints.erase(j + 1);
      }
    }
    ++i;
  }
}

std::set<std::string> Constraints::VariablesUsed() {
  // Returns all the variables appearing in the constraints
  std::set<std::string> ret;
  for (const auto& constraint : constraints) {
    for (const auto& kvp : constraint.poly.getMap()) {
      const auto& key = kvp.first;
      if (key.size()) {  // Do nothing for constant term
        ret.emplace(key);
      }
    }
  }
  return ret;
}

// TODO(T133): Check size of integer programming problem to prevent slowdown
std::tuple<IndexBounds, SimpleConstraints> Constraints::ComputeBounds() {
  auto vars = VariablesUsed();

  // Run the solver for each variable min + max
  bilp::ILPSolver solver;
  IndexBounds out;
  IndexAccess objectives;
  for (const std::string& var : vars) {
    objectives.emplace_back(var);
    objectives.emplace_back(var, -1);
  }
  std::map<IndexPoly, bilp::ILPResult> result = solver.batch_solve(constraints, objectives);
  for (const auto& kvp : result) {
    // ILPResult lists the objective for each requested optimization. Since we
    // used a monomial for each objective, GetNonzeroIndex returns the name of
    // the variable. Then we grab its coefficient to see if we were requesting
    // minimization or maximization
    std::string var = kvp.first.GetNonzeroIndex();
    if (kvp.first[var] == 1) {
      out[var].min = static_cast<int64_t>(kvp.second.obj_val);
    } else if (kvp.first[var] == -1) {
      out[var].max = static_cast<int64_t>(-kvp.second.obj_val);
    } else {
      throw std::runtime_error("Internal error: unexpected ILP objective type");
    }
  }

  // Remove constraints which are implied
  SimpleConstraints remaining;
  for (const auto& constraint : constraints) {
    auto lower = constraint.lowerBound();
    if (!IsImplied(lower, out)) {
      remaining.push_back(lower);
    }
    auto upper = constraint.upperBound();
    if (!IsImplied(upper, out)) {
      remaining.push_back(upper);
    }
  }

  return std::tie(out, remaining);
}

static IndexPoly MakePoly(mlir::Value* value, std::map<std::string, mlir::Value*>* map) {
  if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(value)) {
    auto name = llvm::formatv("x{0}", blockArg->getArgNumber());
    map->emplace(name.str(), value);
    return IndexPoly{name.str()};
  }
  // TODO
  // if (auto op = value->getDefiningOp()) {
  // }
  throw std::runtime_error("Invalid");
}

Contraction::Contraction(ContractionOp op) {
  {
    auto sink = op.getSinkIndexMap();
    auto sinkOp = llvm::cast<AffineSinkIndexMapOp>(sink->getDefiningOp());
    IndexAccess dims;
    for (auto dim : sinkOp.dims()) {
      dims.emplace_back(MakePoly(dim, &argMap));
    }
    accesses.emplace_back(dims);
  }

  for (auto src : op.getSourceIndexMaps()) {
    auto srcOp = llvm::cast<AffineSourceIndexMapOp>(src->getDefiningOp());
    IndexAccess dims;
    for (auto dim : srcOp.dims()) {
      dims.emplace_back(MakePoly(dim, &argMap));
    }
    accesses.emplace_back(dims);
  }
}

Constraints Contraction::GatherConstraints(llvm::ArrayRef<stripe::TensorType> shapes) const {
  // Make the output collection
  Constraints ret;
  // Add all the simple constraints
  for (const auto& constraint : constraints) {
    ret.constraints.push_back(constraint);
  }
  // Sanity check the shapes
  if (shapes.size() != accesses.size()) {
    throw std::runtime_error(
        str(boost::format("Shape mismatch during constraint gathering: %1% vs %2%") % shapes.size() % accesses.size()));
  }
  for (size_t i = 0; i < accesses.size(); i++) {
    ret.AddTensor(accesses[i], shapes[i]);
  }
  std::stable_sort(ret.constraints.begin(), ret.constraints.end(), [](const auto& x, const auto& y) {  //
    return (x.range < y.range);
  });
  return ret;
}

void Contraction::ConstrainIndexVarsToInts() {
  std::vector<math::RangeConstraint> newConstraints;
  const int32_t kBoundWidth = 1000000000;
  for (const auto& var : getIndexVars()) {
    if (!var.empty()) {  // Constant components not used
      newConstraints.emplace_back(RangeConstraint(IndexPoly(var) + kBoundWidth / 2, kBoundWidth));
    }
  }
  constraints = newConstraints;
}

std::set<std::string> Contraction::getIndexVars() const {
  std::set<std::string> vars;
  for (const auto& access : accesses) {
    for (const auto& poly : access) {
      for (const auto& kvp : poly.getMap()) {
        vars.insert(kvp.first);
      }
    }
  }
  // Valdiate that no variables appear only in the constraints
  // Pre-add the 'constant' variable (which is always valid)
  // vars.insert("");
  // for (const auto& cons : constraints) {
  //   for (const auto& kvp : cons.bound.poly.getMap()) {
  //     // If if there is a variable used in a constraint and that variable doesn't
  //     // appear in the list of variables from the tensors, then it's a variable
  //     // that appears only in the constraints, and we need to throw.
  //     if (vars.find(kvp.first) == vars.end()) {
  //       std::ostringstream ErrorStr;
  //       ErrorStr << "Contraction::getIndexAndOutputVars: Variable '" << kvp.first
  //                << "' appears only in constraints of contraction:\n"
  //                << " Tensors:";
  //       for (const auto& tensor : specs) {
  //         ErrorStr << " {";
  //         for (const auto& poly : tensor.spec) {
  //           ErrorStr << poly.toString() << ", ";
  //         }
  //         ErrorStr.seekp(-2, ErrorStr.cur);
  //         ErrorStr << "},";
  //       }
  //       ErrorStr.seekp(-1, ErrorStr.cur);
  //       ErrorStr << "\nConstraints:";
  //       for (const auto& cons_error : constraints) {
  //         ErrorStr << " { Poly: " << cons_error.bound.poly.toString();
  //         ErrorStr << ", Range: " << std::to_string(cons_error.bound.range);
  //         ErrorStr << ", Var: " << cons_error.range << " }";
  //       }
  //       throw std::runtime_error(ErrorStr.str());
  //     }
  //   }
  // }
  // Erase constant
  vars.erase("");
  return vars;
}

static IndexPoly ConvertVariables(     //
    const IndexPoly& in,               //
    llvm::ArrayRef<std::string> vars,  //
    llvm::ArrayRef<IndexPoly> polys) {
  IndexPoly out;
  for (size_t i = 0; i < vars.size(); i++) {
    out += in[vars[i]] * polys[i];
  }
  out += in.constant();
  return out;
}

static IndexPoly ConvertPoly(                       //
    IndexPoly in,                                   //
    const std::map<std::string, IndexPoly>& polys,  //
    bool transform_constant = false) {
  IndexPoly out;
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

void Contraction::ReduceOutputPolynomials(const Constraints& order) {
  // First, we find all of our index variables
  std::set<std::string> indexVars = getIndexVars();

  // Now we construct a set of 'rewrite' variables for each linearly independent
  // output IndexPoly
  math::BasisBuilder basis;
  for (const auto& poly : accesses[0]) {
    // Maybe add it to the equation list
    basis.addEquation(poly);
  }

  // Next, fill in from contraints until we have as many as variables or we run
  // out of options
  for (const auto& constraint : order.constraints) {
    if (basis.dimensions() == indexVars.size()) {
      break;
    }
    basis.addEquation(constraint.poly);
  }
  const IndexAccess& forms = basis.basis();
  if (forms.size() < indexVars.size()) {
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
  math::Matrix matrix;
  std::tie(matrix, std::ignore) = FromPolynomials(forms);
  if (!matrix.invert()) {
    throw std::runtime_error("Attempt to solve indexing equations failed due to singular matrix");
  }

  // Now convert back to equations
  IndexAccess inverses;
  for (size_t i = 0; i < size; i++) {
    IndexPoly poly;
    for (size_t j = 0; j < size; j++) {
      poly += matrix(i, j) * IndexPoly(std::string("v") + std::to_string(j));
    }
    inverses.push_back(poly);
  }

  // Vectorize variables
  std::vector<std::string> vec_idx(indexVars.begin(), indexVars.end());

  // Print new equations
  if (VLOG_IS_ON(3)) {
    VLOG(3) << "In reduce, reduced equations are:";
    for (size_t i = 0; i < size; i++) {
      VLOG(3) << "  " << vec_idx[i] << " == " << to_string(inverses[i]);
    }
  }

  for (auto& access : accesses) {
    for (auto& poly : access) {
      poly = ConvertVariables(poly, vec_idx, inverses);
    }
  }

  for (auto& constraint : constraints) {
    constraint = RangeConstraint(ConvertVariables(constraint.poly, vec_idx, inverses), constraint.range);
  }

  // Make new version of the op by first copying the original
  // Contraction new_op(op.specs.size() - 1);
  // new_op.comb_op = op.comb_op;
  // new_op.agg_op = op.agg_op;
  // for (size_t i = 0; i < op.specs.size(); i++) {
  //   for (size_t j = 0; j < op.specs[i].spec.size(); j++) {
  //     new_op.specs[i].spec.push_back(ConvertVariables(op.specs[i].spec[j], vec_idx, inverses));
  //     new_op.specs[i].id = op.specs[i].id;
  //   }
  // }

  // for (size_t i = 0; i < op.constraints.size(); i++) {
  //   const RangeConstraint& oc = op.constraints[i].bound;
  //   new_op.constraints.push_back(
  //       SymbolicConstraint(RangeConstraint(ConvertVariables(oc.poly, vec_idx, inverses), oc.range)));
  // }
  // return new_op;
}

void Contraction::Defractionalize(const Constraints& order) {
  IndexAccess polys;
  bool has_fract = false;
  std::set<std::string> vars;
  for (const auto& constraint : order.constraints) {
    for (const auto& kvp : constraint.poly.getMap()) {
      if (denominator(kvp.second) != 1) {
        has_fract = true;
      }
      if (kvp.first != "") {
        vars.insert(kvp.first);
      }
    }
    polys.push_back(constraint.poly);
  }
  if (!has_fract) {
    IVLOG(3, "No fractions to Defract")
    return;
  }
  std::vector<std::string> vvars(vars.begin(), vars.end());

  math::Matrix mat;
  math::Vector vec;
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

  math::Matrix p = prod(trans(mat), mat);
  IVLOG(4, "Product Matrix: " << p);
  if (!p.invert()) {
    throw std::runtime_error("Unable to invert Hermite Matrix");
  }
  IVLOG(4, "Inverse Product Matrix: " << p);
  math::Matrix d = prod(mat, p);
  IVLOG(4, "Dual Matrix: " << d);
  IVLOG(3, "Normalized Dual Matrix: " << d);
  if (!HermiteNormalForm(d)) {
    throw std::runtime_error("Unable to perform Hermite Reduction on dual during defractionalization");
  }

  std::vector<RangeConstraint> new_cons;
  std::vector<std::map<Integer, IndexPoly>> mod_polys;
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
    IndexPoly cpoly;
    std::map<Integer, IndexPoly> ipoly;
    for (size_t j = 0; j < split_vec.size(); j++) {
      std::string var = vvars[i] + "_" + std::to_string(j);
      // Add a constraint for non-final cases
      if (j < split_vec.size() - 1) {
        if (split_vec[j + 1] % split_vec[j] != 0) {
          throw std::runtime_error("Unable to remove modulo operations during defractionalization");
        }
        Integer div = split_vec[j + 1] / split_vec[j];
        new_cons.emplace_back(IndexPoly(var), static_cast<int64_t>(div));
      }
      // Add an entry to the polynomial
      cpoly += split_vec[j] * IndexPoly(var);
      // Add the polynomial to a lookup table
      Integer modof = (j + 1 < split_vec.size() ? split_vec[j + 1] : 0);
      ipoly[modof] = cpoly;
    }
    IVLOG(4, "List of polys: " << ipoly);
    mod_polys.push_back(ipoly);
  }

  // Now, make replacements
  std::map<std::string, IndexPoly> replacements;
  for (size_t i = 0; i < d.size2(); i++) {
    IndexPoly poly = d(i, i) * mod_polys[i][0];
    for (size_t j = 0; j < i; j++) {
      poly += d(j, i) * mod_polys[j][denominator(d(j, i) / d(i, i))];
    }
    replacements[vvars[i]] = poly;
  }

  IVLOG(3, "Replacements = " << replacements);
  IVLOG(3, "New Constraints = " << new_cons);

  for (auto& access : accesses) {
    for (auto& poly : access) {
      poly = ConvertPoly(poly, replacements, transform_constant);
    }
  }

  for (auto& constraint : constraints) {
    constraint = RangeConstraint(ConvertPoly(constraint.poly, replacements, transform_constant), constraint.range);
  }
  if (transform_constant) {
    // Add constraint for the constant term if it's being transformed
    constraints.emplace_back(RangeConstraint(ConvertPoly(IndexPoly(1), replacements, transform_constant) - 1, 1));
  }
  for (const auto& constraint : new_cons) {
    constraints.push_back(constraint);
  }

  // Make new version of the op by first copying the original
  // Contraction new_op(op.specs.size() - 1);
  // new_op.comb_op = op.comb_op;
  // new_op.agg_op = op.agg_op;
  // for (size_t i = 0; i < op.specs.size(); i++) {
  //   for (size_t j = 0; j < op.specs[i].spec.size(); j++) {
  //     new_op.specs[i].spec.push_back(ConvertPoly(op.specs[i].spec[j], replacements, transform_constant));
  //     new_op.specs[i].id = op.specs[i].id;
  //   }
  // }

  // for (size_t i = 0; i < op.constraints.size(); i++) {
  //   const RangeConstraint& oc = op.constraints[i].bound;
  //   new_op.constraints.push_back(
  //       SymbolicConstraint(RangeConstraint(ConvertPoly(oc.poly, replacements, transform_constant), oc.range)));
  // }
  // if (transform_constant) {
  //   // Add constraint for the constant term if it's being transformed
  //   new_op.constraints.push_back(SymbolicConstraint(
  //       RangeConstraint(ConvertPoly(IndexPoly(1), replacements, transform_constant) - 1, 1)));
  // }

  // for (const RangeConstraint& c : new_cons) {
  //   new_op.constraints.push_back(SymbolicConstraint(c));
  // }
  // return new_op;
}

bool Contraction::NeedReduce() const {
  for (const auto& poly : accesses[0]) {
    if (poly.getMap().size() > 2 || (poly.getMap().size() == 2 && poly.constant() == 0)) {
      return true;
    }
  }
  return false;
}

std::tuple<IndexBounds, SimpleConstraints> Contraction::ComputeBounds(llvm::ArrayRef<stripe::TensorType> shapes) {
  ConstrainIndexVarsToInts();
  auto constraints = GatherConstraints(shapes);
  // Reduce if needed
  if (NeedReduce()) {
    ReduceOutputPolynomials(constraints);
    constraints = GatherConstraints(shapes);
  }
  constraints.MergeParallelConstraints();
  // Defract if needed (defract does early return if not required)
  Defractionalize(constraints);
  // Gather the constraints from index bounds
  constraints = GatherConstraints(shapes);
  // New parallel constraints might have been introduced by defract; re-merge them
  constraints.MergeParallelConstraints();
  return constraints.ComputeBounds();
}

}  // namespace tile
}  // namespace dialect
}  // namespace pmlc
