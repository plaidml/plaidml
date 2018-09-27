#pragma once

#include <array>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/operators.hpp>

#include "base/util/logging.h"
#include "base/util/printstring.h"
#include "tile/math/bignum.h"

namespace vertexai {
namespace tile {
namespace lang {

// A linear Polynomial of Rational coefficients
class Polynomial : boost::additive<Polynomial>,
                   boost::ring_operators<Polynomial, Rational>,
                   boost::dividable<Polynomial, Rational>,
                   boost::equality_comparable<Polynomial> {
 public:
  // Construct Polynomials
  Polynomial();  // Zero Polynomial
  // clang-format off
  //cppcheck-suppress noExplicitConstructor  // NOLINT
  Polynomial(const Rational& c);  // Constant Polynomial  // NOLINT
  //cppcheck-suppress noExplicitConstructor  // NOLINT
  Polynomial(const std::string& i, const Rational& c = 1);  // Monomial  // NOLINT
  // clang-format on
  Rational operator[](const std::string& var) const;      // Quick coefficent access
  const std::map<std::string, Rational>& getMap() const;  // Get inner map
  bool operator==(const Polynomial& rhs) const;           // Equality
  bool operator<(const Polynomial& rhs) const;            // Lexigraphical order
  Polynomial& operator+=(const Polynomial& rhs);          // Addition
  Polynomial& operator-=(const Polynomial& rhs);          // Subtraction
  Polynomial operator-() const;                           // Unary minus
  Polynomial& operator*=(const Rational& rhs);            // Multiplication by a Rational
  Polynomial& operator/=(const Rational& rhs);            // Division by a rations
  bool isConstant() const { return map_.size() == 0 || (map_.size() == 1 && map_.find("") != map_.end()); }
  Rational constant() const;         // Get the constant part of the Polynomial
  void setConstant(Rational value);  // Set the constant part of the Polynomial to value
  Rational eval(const std::map<std::string, Rational>& values) const;
  // If this/p has no remainder, return this/p, otherwise return 0
  // This works even if p == 0
  Rational tryDivide(const Polynomial& p, bool ignoreConst = false) const;
  // Substitute replacement in for var in this polynomial
  void substitute(const std::string& var, const Polynomial& replacement);
  // If the string has a nonzero coefficient for at least one of its nonconstant
  // indices, it will return the index name of one such index. No promises about
  // which index you'll get. Returns empty string if no index w/ nonconst coeff
  std::string GetNonzeroIndex() const;

  std::string toString() const;  // Pretty-print to string

 private:
  // Map from index -> coefficient
  // Constant offset is a coefficent of empty string
  std::map<std::string, Rational> map_;
};

// Friendly utility to play nice with std::to_string
inline std::string to_string(const Polynomial& p) { return p.toString(); }

// Simple Constraint object, mean poly <= rhs
struct SimpleConstraint {
  SimpleConstraint(const Polynomial& _poly, int64_t _rhs);

  Polynomial poly;  // Polynomial constraints apply to
  int64_t rhs;      // Exclusive upper bound, range [0, upper)
};

// Range Constraint object, means 0 <= poly < upper, and value of poly is an integer
struct RangeConstraint {
  RangeConstraint() = default;
  RangeConstraint(const Polynomial& _poly, int64_t _range);
  bool IsParallel(const RangeConstraint& c);

  Polynomial poly;  // Polynomial constraints apply to
  int64_t range;    // Exclusive upper bound, range [0, upper)

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

inline MAKE_LOGGABLE(Polynomial, c, os) {
  os << to_string(c);
  return os;
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
