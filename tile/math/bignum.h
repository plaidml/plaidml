#pragma once

#include <string>

#include <boost/multiprecision/cpp_int.hpp>

namespace vertexai {
namespace tile {
namespace math {

typedef boost::multiprecision::cpp_int_backend<> IntegerBackend;
typedef boost::multiprecision::rational_adaptor<IntegerBackend> RationalBackend;
typedef boost::multiprecision::number<IntegerBackend, boost::multiprecision::et_off> Integer;
typedef boost::multiprecision::number<RationalBackend, boost::multiprecision::et_off> Rational;

inline std::string to_string(const Integer& x) { return x.str(); }
inline std::string to_string(const Rational& x) { return x.str(); }

// Finds greatest int that is <= x
Integer Floor(const Rational& x);
// Finds smallest int that is >= x
Integer Ceil(const Rational& x);
// Computes the fractional part of x
Rational FracPart(const Rational& x);
// Compute the absolute value
Integer Abs(const Integer& x);
// Compute the absolute value
Rational Abs(const Rational& x);
// Modulo like reduction, that is, find r, 0 <= v < m, such that r = k*m + v for some integer k
Rational Reduce(const Rational& v, const Rational& m);  // NOLINT(runtime/references)
// Compute the extended common denominator over rational.  That is, find a return value r,
// such that a / r is an integer, and b / r is an integer, and r = x*a + y*b
Rational XGCD(const Rational& a, const Rational& b, Integer& x, Integer& y);  // NOLINT(runtime/references)
// Same as above, but for a and b both integers
Integer XGCD(const Integer& a, const Integer& b, Integer& x, Integer& y);  // NOLINT(runtime/references)
// Same as above, but don't bother computing x + y
Rational GCD(const Rational& a, const Rational& b);
// Same as above, but for integers
Integer GCD(const Integer& a, const Integer& b);
// Least common multiple for integers
Integer LCM(const Integer& a, const Integer& b);
// Least of 2 numbers
Integer Min(const Integer& a, const Integer& b);
Rational Min(const Rational& a, const Rational& b);
// Largest of 2 numbers
Integer Max(const Integer& a, const Integer& b);
Rational Max(const Rational& a, const Rational& b);
// Returns the integer quotient of dividing a by b, w/ rational remainder 0 <= r < b
Integer RatDiv(const Rational& a, const Rational& b, Rational& r);  // NOLINT(runtime/references)

}  // namespace math
}  // namespace tile
}  // namespace vertexai
