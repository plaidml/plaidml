#include "pmlc/util/math/polynomial.h"

#include <limits>

#include "llvm/Support/FormatVariadic.h"

namespace pmlc::util::math {

namespace {

Rational UnifiedOffset(const Rational& c1, const Rational& c2, const Integer& n1, const Integer& n2) {
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

RangeConstraint IntersectParallelConstraintPairInner(  //
    const RangeConstraint& constraint1,                //
    const RangeConstraint& constraint2) {
  // The primary algorithm for combining two parallel constraints into one. See
  // merge-parallel.tex in /tile/lang for more details.
  // Assumes the RangeConstraints have been validated to have positive ranges, _or_ that
  // for constraint2 specifically it represents a one sided constraint by having an
  // infinite range parameter represented as a range value of -1
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
  Integer lower_bound;
  Integer upper_bound;
  if (constraint2.range > 0) {
    Rational q2_low = math::Min(-c2 / n2, (constraint2.range - 1 - c2) / n2);
    Rational q2_hi = math::Max(-c2 / n2, (constraint2.range - 1 - c2) / n2);
    lower_bound = math::Max(math::Ceil(q1_low + d), math::Ceil(q2_low + d));
    upper_bound = math::Min(math::Floor(q1_hi + d), math::Floor(q2_hi + d));
  } else if (constraint2.range == -1) {
    // Same calculations of lower/upper_bound as above, but with constraint2.range == infinity
    Rational q2_low = -c2 / n2;
    lower_bound = math::Max(math::Ceil(q1_low + d), math::Ceil(q2_low + d));
    upper_bound = math::Floor(q1_hi + d);
  } else {
    throw std::runtime_error("Given constraint with empty range in IntersectParallelConstraintPair: " +
                             to_string(constraint2));
  }
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
  Polynomial<Rational> p(constraint1.poly / n1);
  p.setConstant(merged_offset);
  return RangeConstraint(p, r);
}

}  // namespace

template <typename T>
Polynomial<T>::Polynomial() {}

template <typename T>
Polynomial<T>::Polynomial(const T& c) : Polynomial<T>("", c) {}

template <typename T>
Polynomial<T>::Polynomial(const std::string& i, const T& c) {
  if (c) {
    map_[i] = c;
  }
}

template <typename T>
T Polynomial<T>::eval(const std::map<std::string, T>& values) const {
  T res = 0;
  for (const auto& kvp : map_) {
    if (kvp.first == "") {
      res += kvp.second;
    } else if (values.find(kvp.first) != values.end()) {
      res += kvp.second * values.at(kvp.first);
    } else {
      throw std::runtime_error(
          llvm::formatv("Failed to find value for {0}, when evaluating {1}", kvp.first, toString()));
    }
  }
  return res;
}

template <typename T>
Polynomial<T> Polynomial<T>::partial_eval(const std::map<std::string, T>& values) const {
  Polynomial<T> r = *this;
  T off = 0;
  for (const auto& kvp : values) {
    off += get(kvp.first) * kvp.second;
    r.map_.erase(kvp.first);
  }
  r += off;
  return r;
}

template <typename T>
T Polynomial<T>::operator[](const std::string& var) const {
  auto it = map_.find(var);
  if (it == map_.end()) {
    return 0;
  }
  return it->second;
}

template <typename T>
const std::map<std::string, T>& Polynomial<T>::getMap() const {
  return map_;
}

template <typename T>
std::map<std::string, T>& Polynomial<T>::mutateMap() {
  return map_;
}

template <typename T>
Polynomial<T>& Polynomial<T>::operator+=(const Polynomial<T>& rhs) {
  for (const auto& kvp : rhs.map_) {
    T new_val = (map_[kvp.first] += kvp.second);
    if (new_val == 0) {
      map_.erase(kvp.first);
    }
  }
  return *this;
}

template <typename T>
bool Polynomial<T>::operator==(const Polynomial<T>& rhs) const {
  return map_ == rhs.map_;
}

template <typename T>
bool Polynomial<T>::operator<(const Polynomial<T>& rhs) const {
  return map_ < rhs.map_;
}

template <typename T>
Polynomial<T>& Polynomial<T>::operator-=(const Polynomial<T>& rhs) {
  return *this += -1 * rhs;
}

template <typename T>
Polynomial<T> Polynomial<T>::operator-() const {
  return -1 * (*this);
}

template <typename T>
Polynomial<T>& Polynomial<T>::operator*=(const T& rhs) {
  if (rhs == 0) {
    map_.clear();
  } else {
    for (auto& kvp : map_) {
      kvp.second *= rhs;
    }
  }
  return *this;
}

template <typename T>
Polynomial<T>& Polynomial<T>::operator/=(const T& rhs) {
  return *this *= (1 / rhs);
}

template <typename T>
T Polynomial<T>::constant() const {
  auto it = map_.find("");
  return (it == map_.end() ? 0 : it->second);
}

template <typename T>
void Polynomial<T>::setConstant(T value) {
  if (value == T(0)) {
    map_.erase("");
  } else {
    map_[""] = value;
  }
}

template <typename T>
T Polynomial<T>::tryDivide(const Polynomial<T>& p, bool ignoreConst) const {
  auto it = p.map_.begin();
  if (ignoreConst && it != p.map_.end() && it->first == "") {
    it++;
  }
  T val = 0;
  for (const auto& kvp : map_) {
    if (ignoreConst && kvp.first == "") {
      continue;
    }
    if (it == p.map_.end() || it->first != kvp.first) {
      return 0;  // Indexes don't exactly line up, fail
    }
    T div = kvp.second / it->second;
    if (val != 0 && div != val) {
      return 0;  // They don't all divide by the same number
    }
    val = div;
    it++;
  }
  if (it != p.map_.end()) {
    return 0;
  }
  return val;
}

template <typename T>
void Polynomial<T>::substitute(const std::string& var, const Polynomial<T>& replacement) {
  if (map_.count(var) == 0) {
    // If var isn't in this polynomial, nothing needs to be done
    return;
  }
  T coeff = map_[var];
  map_.erase(var);
  (*this) += coeff * replacement;
}

template <typename T>
void Polynomial<T>::substitute(const std::map<std::string, Polynomial<T>>& replacements) {
  Polynomial result;
  for (const auto& name_value : map_) {
    auto replacement = replacements.find(name_value.first);
    if (replacement == replacements.end()) {
      result += Polynomial{name_value.first, name_value.second};
      continue;
    }
    result += replacement->second * name_value.second;
  }
  map_.swap(result.map_);
}

template <typename T>
void Polynomial<T>::substitute(const std::string& var, const T& replacement) {
  substitute(var, Polynomial<T>(replacement));
}

template <typename T>
Polynomial<T> Polynomial<T>::sym_eval(const std::map<std::string, Polynomial> values) const {
  Polynomial<T> out;
  for (const auto& kvp : map_) {
    if (kvp.first.empty()) {
      out += Polynomial<T>(kvp.second);
    } else {
      out += values.at(kvp.first) * kvp.second;
    }
  }
  return out;
}

template <typename T>
std::string Polynomial<T>::GetNonzeroIndex() const {
  // Returns a nonconstant nonzero index, if one exists; otherwise returns empty string
  for (const auto& kvp : map_) {
    if (!(kvp.first.empty()) && kvp.second != 0) return kvp.first;
  }

  // No nonconstant index has a nonzero coefficient
  return std::string();
}

template <typename T>
T Polynomial<T>::get(const std::string& name) const {
  auto it = map_.find(name);
  if (it == map_.end()) {
    return T();
  }
  return it->second;
}

int64_t abs_value(int64_t value) { return std::llabs(value); }

Rational abs_value(Rational value) { return abs(value); }

template <typename T>
std::string Polynomial<T>::toString() const {
  std::stringstream ss;
  if (map_.size() == 0) {
    return "0";
  }
  bool first = true;
  for (const auto& kvp : map_) {
    if (first) {
      if (kvp.second < 0) {
        ss << "-";
      }
      first = false;
    } else {
      if (kvp.second > 0) {
        ss << " + ";
      } else {
        ss << " - ";
      }
    }
    auto value = abs_value(kvp.second);
    if (value != 1 || kvp.first == "") {
      ss << value;
      if (kvp.first != "") {
        ss << "*";
      }
    }
    ss << kvp.first;
  }
  return ss.str();
}

template class Polynomial<Rational>;
template class Polynomial<int64_t>;

SimpleConstraint::SimpleConstraint(const Polynomial<Rational>& poly, int64_t rhs) : poly(poly), rhs(rhs) {}

RangeConstraint::RangeConstraint(const Polynomial<Rational>& poly, int64_t range) : poly(poly), range(range) {}

bool RangeConstraint::IsParallel(const RangeConstraint& c) {
  if (this->poly.tryDivide(c.poly, true) != 0) return true;
  return false;
}

SimpleConstraint RangeConstraint::lowerBound() const { return SimpleConstraint(-poly, 0); }

SimpleConstraint RangeConstraint::upperBound() const { return SimpleConstraint(poly, range - 1); }

bool IsImplied(const SimpleConstraint& constraint, const IndexBounds& bounds) {
  auto worst = constraint.poly.constant();
  for (const auto& [key, value] : constraint.poly.getMap()) {
    if (key.empty()) {
      continue;
    }
    if (value < 0) {
      worst += value * bounds.find(key)->second.min;
    } else {
      worst += value * bounds.find(key)->second.max;
    }
  }
  return (worst <= constraint.rhs);
}

RangeConstraint IntersectParallelConstraintPair(  //
    const RangeConstraint& constraint1,           //
    const SimpleConstraint& constraint2) {
  // Combines two parallel constraints into one
  IVLOG(5, "Merging the parallel constraints " << constraint1 << ", " << constraint2);
  if (constraint1.range <= 0) {
    throw std::runtime_error("Given constraint with empty range in IntersectParallelConstraintPair: " +
                             to_string(constraint1));
  }
  return IntersectParallelConstraintPairInner(constraint1, RangeConstraint(constraint2.rhs - constraint2.poly, -1));
}

RangeConstraint IntersectParallelConstraintPair(  //
    const RangeConstraint& constraint1,           //
    const RangeConstraint& constraint2) {
  // Combines two parallel constraints into one
  IVLOG(5, "Merging the parallel constraints " << constraint1 << ", " << constraint2);
  if (constraint1.range <= 0) {
    throw std::runtime_error("Given constraint with empty range in IntersectParallelConstraintPair: " +
                             to_string(constraint1));
  }
  if (constraint2.range <= 0) {
    throw std::runtime_error("Given constraint with empty range in IntersectParallelConstraintPair: " +
                             to_string(constraint2));
  }
  return IntersectParallelConstraintPairInner(constraint1, constraint2);
}

RangeConstraint IntersectOpposedSimpleConstraints(  //
    const SimpleConstraint& constraint1,            //
    const SimpleConstraint& constraint2) {
  // Combines two SimpleConstraints which are parallel and bounding in opposite directions into
  // a single equivalent RangeConstraint
  IVLOG(6, "Merging Opposed SimpleConstraints: " << constraint1 << " and " << constraint2);
  Rational ratio = constraint1.poly.tryDivide(constraint2.poly, true);
  if (ratio >= 0) {
    throw std::invalid_argument(
        "Parameters of IntersectOpposedSimpleConstraints must be parallel and in opposite directions");
  }
  // We know constraint1.poly <= constraint1.rhs and also constraint2.poly <= constraint2.rhs
  // Could rewrite as 0 <= -constraint1.poly + constraint1.rhs and also 0 <= -constraint2.poly + constraint2.rhs
  // Rewrite: constraint1.poly = p1 + c1 (where p1 has no constant part), constraint1.rhs = rhs1; similarly for 2
  // So 0 <= -p1 - c1 + rhs1 and also 0 <= -p2 - c2 + rhs2
  // and since a = ratio is negative, and p2 * a = p1, get
  //   0 >= -a * p2 - a * c2 + a * rhs2
  //   0 >= -p1 - a * c2 + a * rhs2
  //   -c1 + rhs1 >= -p1 - a * c2 + a * rhs2 - c1 + rhs1
  //   -c1 + rhs1 + a * c2 - a * rhs2 >= -p1 -c1 + rhs1
  // So, merge into
  //   0 <= -p1 - c1 + rhs1 <= -c1 + rhs1 + a * c2 - a * rhs2
  // i.e.
  //   0 <= -p1 - c1 + rhs1 < -c1 + rhs1 + a * c2 - a * rhs2 + 1
  // (noting that this is just for the bounds, not the integrality, and noting that the right hand bound will be
  // slightly too generous if the RHS is not an integer; see discussion below for why this is ok)
  // And so use this to make a range constraint from constraint1 and then call to IntersectParallelConstraintPair
  auto merged_poly1 = -constraint1.poly + constraint1.rhs;
  auto merged_const1 = -constraint1.poly.constant() + constraint1.rhs;
  auto merged_const2 = -constraint2.poly.constant() + constraint2.rhs;
  // Note that if the range is too generous the IntersectParallel call will fix it. So, instead of figuring out
  // whether there is a fractional offset in the new form and determining the precise range, use Ceil and Floor
  // to get an overapproximation of the range.
  // This approach may duplicate a bit of work, but I think this is too tiny for that to matter for overall perf
  auto range = math::Ceil(merged_const1) - math::Floor(ratio * merged_const2) + 1;
  if (range > INT64_MAX) {
    throw std::out_of_range("Bound range in IntersectOpposedSimpleConstraints overflows int64.");
  }
  int64_t r = (int64_t)range;
  // IVLOG(6, "Ratio: " << ratio << "; merged_poly1: " << merged_poly1 << "; merged_const1: " << merged_const1
  //                    << "; merged_const2: " << merged_const2);
  // IVLOG(6, "Range (& formula): " << range << " = ceil(" << merged_const1 << ") - floor(" << ratio << " * "
  //                                << merged_const2 << ")");
  return IntersectParallelConstraintPair(RangeConstraint(merged_poly1, r), constraint2);
}

}  // namespace pmlc::util::math
