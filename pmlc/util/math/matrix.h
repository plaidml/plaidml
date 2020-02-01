#pragma once

#include <string>
#include <tuple>
#include <vector>

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>

#include "pmlc/util/math/polynomial.h"

namespace pmlc::util::math {

class Matrix : public boost::numeric::ublas::matrix<Rational> {
public:
  // Constructors (inherit all from ublas)
  Matrix() : boost::numeric::ublas::matrix<Rational>() {}
  Matrix(size_type size1, size_type size2)
      : boost::numeric::ublas::matrix<Rational>(size1, size2) {}
  Matrix(size_type size1, size_type size2, const value_type &init)
      : boost::numeric::ublas::matrix<Rational>(size1, size2, init) {}
  Matrix(size_type size1, size_type size2, const array_type &data)
      : boost::numeric::ublas::matrix<Rational>(size1, size2, data) {}
  // These one-param ctors are not explicit b/c boost does not make them
  // explicit; so presumably implicit conversion is an intended use
  Matrix(const boost::numeric::ublas::matrix<Rational>
             &m) // NOLINT(runtime/explicit)
      : boost::numeric::ublas::matrix<Rational>(m) {}
  template <class AE>
  Matrix(const boost::numeric::ublas::matrix_expression<AE>
             &ae) // NOLINT(runtime/explicit)
      : boost::numeric::ublas::matrix<Rational>(ae) {}

  // Row ops
  void swapRows(size_t r, size_t s);
  void multRow(size_t r, Rational multiplier);
  void addRowMultToRow(size_t dest_row, size_t src_row,
                       const Rational &multiplier);
  void makePivotAt(size_t row, size_t col);

  // Performs a rational matrix inversion, return false if singular
  bool invert();

  std::string toString() const; // Pretty-print to string

  bool operator==(const Matrix &m);
};

// Play nice with std::to_string
inline std::string to_string(const Matrix &m) { return m.toString(); }

typedef boost::numeric::ublas::vector<Rational> Vector;
typedef boost::numeric::ublas::identity_matrix<Rational> IdentityMatrix;

using boost::numeric::ublas::prod;
using boost::numeric::ublas::trans;

// Matrix literals to make things easier
Vector VectorLit(const std::vector<Rational> &vec);
Matrix MatrixLit(const std::vector<std::vector<Rational>> &vecs);

// Equality operators
bool operator==(const Vector &a, const Vector &b);
bool operator==(const Matrix &a, const Matrix &b);

// Returns a matrix of polys.size() rows and with one column for each
// variable in polys (in lexigraphical order).  Also, returns a vector of
// polys.size() containing the constants for each polynomial
std::tuple<Matrix, Vector>
FromPolynomials(const std::vector<Polynomial<Rational>> &polys);

// Convert matrix + vector of offsets to Hermite Normal Form
bool HermiteNormalForm(Matrix &m); // NOLINT(runtime/references)

} // namespace pmlc::util::math
