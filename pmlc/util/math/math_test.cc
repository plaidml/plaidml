#include "pmlc/util/logging.h"
#include "pmlc/util/math/basis.h"
#include "pmlc/util/math/bignum.h"
#include "pmlc/util/math/matrix.h"
#include "pmlc/util/math/polynomial.h"

namespace pmlc::util::math {

// TEST_CASE("Ceil", "[lattice]") {
//   REQUIRE(Ceil(Rational(5, 3)) == 2);
//   REQUIRE(Ceil(Rational(6, 3)) == 2);
//   REQUIRE(Ceil(Rational(7, 3)) == 3);
//   REQUIRE(Ceil(Rational(0, 3)) == 0);
//   REQUIRE(Ceil(Rational(-1, 3)) == 0);
//   REQUIRE(Ceil(Rational(-3, 3)) == -1);
//   REQUIRE(Ceil(Rational(-4, 3)) == -1);
// }

// TEST_CASE("Floor", "[lattice]") {
//   REQUIRE(Floor(Rational(5, 3)) == 1);
//   REQUIRE(Floor(Rational(7, 3)) == 2);
//   REQUIRE(Floor(Rational(0, 3)) == 0);
//   REQUIRE(Floor(Rational(-1, 3)) == -1);
//   REQUIRE(Floor(Rational(-3, 3)) == -1);
//   REQUIRE(Floor(Rational(-4, 3)) == -2);
// }

// TEST_CASE("Reduce", "[lattice]") {
//   REQUIRE(Reduce(5, 3) == 2);
//   REQUIRE(Reduce(7, 3) == 1);
//   REQUIRE(Reduce(0, 3) == 0);
//   REQUIRE(Reduce(-1, 3) == 2);
//   REQUIRE(Reduce(-3, 3) == 0);
//   REQUIRE(Reduce(-4, 3) == 2);
//   REQUIRE(Reduce(Rational(1, 2), 1) == Rational(1, 2));
//   REQUIRE(Reduce(Rational(13, 4), Rational(1, 2)) == Rational(1, 4));
//   REQUIRE(Reduce(Rational(13, 4), Rational(1, 3)) == Rational(1, 4));
//   REQUIRE(Reduce(Rational(13, 4), Rational(1, 5)) == Rational(1, 20));
// }

// static void ValidateXGCD(const Rational& a, const Rational& b) {
//   Integer x, y;
//   Rational o = XGCD(a, b, x, y);
//   REQUIRE(denominator(a / o) == 1);
//   REQUIRE(denominator(b / o) == 1);
//   REQUIRE(x * a + y * b == o);
//   o.str();
// }

// TEST_CASE("XGCD", "[lattice]") {
//   ValidateXGCD(5, 1);
//   ValidateXGCD(1, 5);
//   ValidateXGCD(-5, 1);
//   ValidateXGCD(1, -5);
//   ValidateXGCD(5, -1);
//   ValidateXGCD(-1, 5);
//   ValidateXGCD(100, 15);
//   ValidateXGCD(Rational(25, 6), Rational(15, 8));
//   ValidateXGCD(Rational(15, 8), Rational(25, 6));
//   ValidateXGCD(Rational(-15, 8), Rational(25, 6));
//   ValidateXGCD(Rational(15, 8), Rational(-25, 6));
//   ValidateXGCD(Rational(-15, 8), Rational(-25, 6));
// }

// TEST_CASE("From Poly Test", "[matrix][fromPoly]") {
//   Polynomial<Rational> x("x"), y("y"), z("z");
//   Matrix m;
//   Vector v;
//   std::tie(m, v) = FromPolynomials({{x + 5}, {3 * y - 2 * z}, {-x + z - 2}});
//   REQUIRE(m(0, 0) == 1);
//   REQUIRE(m(0, 1) == 0);
//   REQUIRE(m(0, 2) == 0);
//   REQUIRE(v(0) == 5);
//   REQUIRE(m(1, 0) == 0);
//   REQUIRE(m(1, 1) == 3);
//   REQUIRE(m(1, 2) == -2);
//   REQUIRE(v(1) == 0);
//   REQUIRE(m(2, 0) == -1);
//   REQUIRE(m(2, 1) == 0);
//   REQUIRE(m(2, 2) == 1);
//   REQUIRE(v(2) == -2);
// }

// TEST_CASE("Inversion Test (Regular)", "[matrix][invert]") {
//   Matrix m = MatrixLit({{1, 2, 3}, {4, 0, 6}, {7, 8, 9}});
//   m.invert();
//   REQUIRE(m(0, 0) == Rational(-4, 5));
//   REQUIRE(m(0, 1) == Rational(1, 10));
//   REQUIRE(m(0, 2) == Rational(1, 5));
//   REQUIRE(m(1, 0) == Rational(1, 10));
//   REQUIRE(m(1, 1) == Rational(-1, 5));
//   REQUIRE(m(1, 2) == Rational(1, 10));
//   REQUIRE(m(2, 0) == Rational(8, 15));
//   REQUIRE(m(2, 1) == Rational(1, 10));
//   REQUIRE(m(2, 2) == Rational(-2, 15));
// }

// TEST_CASE("Inversion Test (Singular)", "[matrix][invert]") {
//   Matrix m = MatrixLit({{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
//   REQUIRE(m.invert() == false);
// }

// TEST_CASE("Basic Polynomial<Rational> compilation", "[poly]") {
//   Polynomial<Rational> i("i"), j("j"), k("k");
//   Polynomial<Rational> x = 3 * i - j + 17;
//   Polynomial<Rational> y = -x / 3 + k;
//   REQUIRE(to_string(y) == "-17/3 - i + 1/3*j + k");
// }

// TEST_CASE("Basic Polynomial<Rational> evaluation", "[poly]") {
//   Polynomial<Rational> a0("a0"), a1("a1");
//   Polynomial<Rational> r = a0 + 3 * a1 + 1;
//   REQUIRE(to_string(r) == "1 + a0 + 3*a1");
//   REQUIRE(r.eval({{"a0", 5}, {"a1", 9}}) == 33);
// }

// TEST_CASE("IntersectParallelConstraintPair", "[poly]") {
//   Polynomial<Rational> i("i"), j("j");
//   RangeConstraint c1{2 * i + j + 1, 8};
//   RangeConstraint c2{-4 * i - 2 * j - 2, 2};
//   auto intersection = IntersectParallelConstraintPair(c1, c2);
//   REQUIRE(to_string(intersection) == "0 <= -1 - 2*i - j < 1");
// }

// TEST_CASE("IntersectOpposedSimpleConstraintsMultivar", "[poly]") {
//   Polynomial<Rational> i("i"), j("j");
//   SimpleConstraint c1{2 * i + j + 1, 4};
//   SimpleConstraint c2{-6 * i - 3 * j, 2};
//   auto intersection = IntersectOpposedSimpleConstraints(c1, c2);
//   REQUIRE(to_string(intersection) == "0 <= 2*i + j < 4");
// }

// TEST_CASE("IntersectOpposedSimpleConstraints", "[poly]") {
//   Polynomial<Rational> x("x");
//   SimpleConstraint c1{x + 1, 4};
//   SimpleConstraint c2{-3 * x, 2};
//   auto intersection = IntersectOpposedSimpleConstraints(c1, c2);
//   REQUIRE(to_string(intersection) == "0 <= x < 4");
// }

// TEST_CASE("IntersectOpposedSimpleBasic", "[poly]") {
//   Polynomial<Rational> x("x");
//   SimpleConstraint c1{x, 10};
//   SimpleConstraint c2{-x, 0};
//   auto intersection = IntersectOpposedSimpleConstraints(c1, c2);
//   REQUIRE(to_string(intersection) == "0 <= x < 11");
// }

// TEST_CASE("HNFMatrix", "[hnf]") {
//   Matrix m = MatrixLit({{0, Rational(1, 2)}, {Rational(1, 2), Rational(1, 2)}, {1, 0}});
//   bool r = HermiteNormalForm(m);
//   REQUIRE(r == true);
// }
// TEST_CASE("HNFMatrix Matches Mathematica", "[hnf]") {
//   Matrix m = MatrixLit({{2, 6, -3, 2}, {2, 6, -3, 2}, {2, 18, 3, 2}, {-6, -3, -6, 2}, {8, 9, -3, 4}, {4, 9, -3, 2}});
//   bool r = HermiteNormalForm(m);
//   REQUIRE(r == true);
//   Matrix correct = MatrixLit({{2, 0, 3, 0}, {0, 3, 3, 0}, {0, 0, 6, 0}, {0, 0, 0, 2}, {0, 0, 0, 0}, {0, 0, 0, 0}});
//   bool match = (m == correct);
//   REQUIRE(match);
// }

// TEST_CASE("HNFMatrix Low Rank ", "[hnf]") {
//   Matrix m = MatrixLit({{Rational(1, 5), -Rational(2, 5)}, {-Rational(2, 5), Rational(4, 5)}});
//   bool r = HermiteNormalForm(m);
//   REQUIRE(r == true);
//   REQUIRE(m(1, 0) == 0);
//   REQUIRE(m(1, 1) == 0);
// }

// TEST_CASE("Basis reduction test", "[basis]") {
//   Polynomial<Rational> x("x"), y("y"), i("i"), j("j");
//   BasisBuilder bb;
//   bool r;
//   r = bb.addEquation(2 * x + 3 * i);
//   REQUIRE(r == true);
//   r = bb.addEquation(-x);
//   REQUIRE(r == true);
//   r = bb.addEquation(i);
//   REQUIRE(r == false);
// }

}  // namespace pmlc::util::math
