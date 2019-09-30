#pragma once

#include <array>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/operators.hpp>

#include "base/util/logging.h"
#include "tile/math/bignum.h"

namespace vertexai {
namespace tile {
namespace math {

// A linear Polynomial<Rational> of Rational coefficients
template <typename T>
class Polynomial : boost::additive<Polynomial<T>>,
                   boost::ring_operators<Polynomial<T>, T>,
                   boost::dividable<Polynomial<T>, T>,
                   boost::equality_comparable<Polynomial<T>> {
 public:
  // Construct Polynomial<T>s
  Polynomial();  // Zero Polynomial
  // clang-format off
  //cppcheck-suppress noExplicitConstructor  // NOLINT
  Polynomial(const T& c);  // Constant Polynomial<T>  // NOLINT
  //cppcheck-suppress noExplicitConstructor  // NOLINT
  Polynomial(const std::string& i, const T& c = 1);  // Monomial  // NOLINT
  // clang-format on
  T operator[](const std::string& var) const;      // Quick coefficent access
  const std::map<std::string, T>& getMap() const;  // Get inner map
  std::map<std::string, T>& mutateMap();           // Get inner map for editing
  bool operator==(const Polynomial& rhs) const;    // Equality
  bool operator<(const Polynomial& rhs) const;     // Lexigraphical order
  Polynomial& operator+=(const Polynomial& rhs);   // Addition
  Polynomial& operator-=(const Polynomial& rhs);   // Subtraction
  Polynomial operator-() const;                    // Unary minus
  Polynomial& operator*=(const T& rhs);            // Multiplication by a T
  Polynomial& operator/=(const T& rhs);            // Division by a rations
  bool isConstant() const { return map_.size() == 0 || (map_.size() == 1 && map_.find("") != map_.end()); }
  T constant() const;         // Get the constant part of the Polynomial<T>
  void setConstant(T value);  // Set the constant part of the Polynomial<T> to value
  T eval(const std::map<std::string, T>& values) const;
  Polynomial partial_eval(const std::map<std::string, T>& values) const;
  // If this/p has no remainder, return this/p, otherwise return 0
  // This works even if p == 0
  T tryDivide(const Polynomial& p, bool ignoreConst = false) const;
  // Substitute replacement in for var in this polynomial
  void substitute(const std::string& var, const Polynomial<T>& replacement);
  void substitute(const std::map<std::string, Polynomial<T>>& replacements);
  void substitute(const std::string& var, const T& replacement);
  // Symbolically evaluate a polynomial
  Polynomial sym_eval(const std::map<std::string, Polynomial<T>> values) const;
  // If the string has a nonzero coefficient for at least one of its nonconstant
  // indices, it will return the index name of one such index. No promises about
  // which index you'll get. Returns empty string if no index w/ nonconst coeff
  std::string GetNonzeroIndex() const;

  T get(const std::string& name) const;

  std::string toString() const;  // Pretty-print to string

 private:
  // Map from index -> coefficient
  // Constant offset is a coefficent of empty string
  std::map<std::string, T> map_;
};

extern template class Polynomial<Rational>;
extern template class Polynomial<int64_t>;

using Affine = math::Polynomial<int64_t>;

// Friendly utility to play nice with std::to_string
inline std::string to_string(const Polynomial<Rational>& p) { return p.toString(); }
inline std::string to_string(const Polynomial<int64_t>& p) { return p.toString(); }

// Simple Constraint object, mean poly <= rhs
struct SimpleConstraint {
  SimpleConstraint(const Polynomial<Rational>& poly, int64_t rhs);

  Polynomial<Rational> poly;  // Polynomial<Rational> constraints apply to
  int64_t rhs;                // poly <= rhs
};

// Range Constraint object, means 0 <= poly < upper, and value of poly is an integer
struct RangeConstraint {
  RangeConstraint() = default;
  RangeConstraint(const Polynomial<Rational>& poly, int64_t range);
  bool IsParallel(const RangeConstraint& c);

  Polynomial<Rational> poly;  // Polynomial<Rational> constraints apply to
  int64_t range;              // Exclusive upper bound, range [0, upper)

  SimpleConstraint lowerBound() const;
  SimpleConstraint upperBound() const;
};

inline std::string to_string(const RangeConstraint& c) {
  return "0 <= " + to_string(c.poly) + " < " + std::to_string(c.range);
}

inline MAKE_LOGGABLE(RangeConstraint, c, os) {
  os << to_string(c);
  return os;
}

inline MAKE_LOGGABLE(SimpleConstraint, c, os) {
  os << to_string(c.poly) << " <= " << c.rhs;
  return os;
}

inline MAKE_LOGGABLE(Polynomial<Rational>, c, os) {
  os << to_string(c);
  return os;
}

}  // namespace math
}  // namespace tile
}  // namespace vertexai
