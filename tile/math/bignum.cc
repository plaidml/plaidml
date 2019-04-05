
#include "tile/math/bignum.h"

#include <boost/math/common_factor_rt.hpp>

namespace vertexai {
namespace tile {
namespace math {

Integer Floor(const Rational& x) {
  if (x < 0) {
    return (numerator(x) - denominator(x) + 1) / denominator(x);
  } else {
    return numerator(x) / denominator(x);
  }
}

Integer Ceil(const Rational& x) { return Floor(Rational(numerator(x) - 1, denominator(x))) + 1; }

int ToInteger(const Rational& x) {
  if (Floor(x) != Ceil(x)) {
    throw std::runtime_error("Non-integer rational.");
  }
  return static_cast<int>(Floor(x));
}

Rational FracPart(const Rational& x) { return x - Floor(x); }

Integer Abs(const Integer& x) {
  if (x < 0) {
    return -x;
  }
  return x;
}

Rational Abs(const Rational& x) {
  if (x < 0) {
    return -x;
  }
  return x;
}

Rational Reduce(const Rational& v, const Rational& m) { return v - Floor(v / m) * m; }

Integer XGCD(const Integer& a, const Integer& b, Integer& x, Integer& y) {  // NOLINT(runtime/references)
  if (b == 0) {
    x = 1;
    y = 0;
    return a;
  }

  Integer x1;
  Integer gcd = XGCD(b, a % b, x1, x);
  y = x1 - (a / b) * x;

  if (gcd < 0) {
    gcd *= -1;
    x *= -1;
    y *= -1;
  }

  return gcd;
}

Rational XGCD(const Rational& a, const Rational& b, Integer& x, Integer& y) {  // NOLINT(runtime/references)
  Integer m = boost::math::lcm(denominator(a), denominator(b));
  Rational o;
  o = Rational(XGCD(numerator(a * m), numerator(b * m), x, y), m);

  if (o < 0) {
    o *= -1;
    x *= -1;
    y *= -1;
  }
  return o;
}

Rational GCD(const Rational& a, const Rational& b) {
  Integer m = boost::math::lcm(denominator(a), denominator(b));
  Integer g = boost::math::gcd(numerator(a * m), numerator(b * m));
  return Rational(g, m);
}

Integer GCD(const Integer& a, const Integer& b) { return boost::math::gcd(a, b); }

Integer LCM(const Integer& a, const Integer& b) { return boost::math::lcm(a, b); }

Integer Min(const Integer& a, const Integer& b) {
  if (a < b)
    return a;
  else
    return b;
}

Rational Min(const Rational& a, const Rational& b) {
  if (a < b)
    return a;
  else
    return b;
}

Integer Max(const Integer& a, const Integer& b) {
  if (a < b)
    return b;
  else
    return a;
}

Rational Max(const Rational& a, const Rational& b) {
  if (a < b)
    return b;
  else
    return a;
}

Integer RatDiv(const Rational& a, const Rational& b, Rational& r) {  // NOLINT(runtime/references)
  Integer q = Floor(a / b);
  r = a - q * b;
  return q;
}

}  // namespace math
}  // namespace tile
}  // namespace vertexai
