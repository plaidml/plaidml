#pragma once

#include <map>
#include <string>
#include <vector>

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>

#include "tile/math/matrix.h"

namespace vertexai {
namespace tile {
namespace lang {
namespace bilp {

using boost::numeric::ublas::range;

typedef std::map<size_t, size_t> RowToColLookup;

class Tableau {
 public:
  // Construct a Tableau initialized to a matrix
  Tableau(const Matrix& m, const std::vector<std::string>& var_names, const std::vector<size_t>* opposites = nullptr);
  // Construct an empty Tableau of dimension size1 x size2
  Tableau(Matrix::size_type size1, Matrix::size_type size2, const std::vector<std::string>& var_names,
          const std::vector<size_t>* opposites = nullptr);
  // Accessors
  // Don't use resize or swap on mat(); the other parts of Tableau won't recognize this and will fail
  const Matrix& mat() const { return matrix_; }
  Matrix& mat() { return matrix_; }
  RowToColLookup basicVars() const;
  std::vector<std::string> varNames() const;
  const std::vector<size_t>& getOpposites() const { return opposites_; }
  // Given column of positive part, return negative part col; and vis-versa
  size_t getOppositePart(size_t var) const { return opposites_[var]; }

  // Optimize Tableau via simplex algorithm
  bool makeOptimal(bool already_canonical = false);
  // Put Tableau in Canonical Form
  // (i.e. contains permuted identity submatrix, last column nonnegative except possibly objective)
  bool convertToCanonicalForm();
  // Find the columns that are basic variables in the current matrix
  void selectBasicVars();
  // Make objective == 0 at each basic variable via row ops
  void priceOut();

  // Solution reporting functions. These report based on the current state of the
  // tableau, so if it hasn't been solved they are likely meaningless
  // Returns the coefficients of the solution in the order of varNames()
  std::vector<Rational> getSymbolicSolution() const;
  // Returns the minimal value the objective can take in the feasible region
  Rational reportObjectiveValue() const;

 protected:
  Matrix matrix_;

 private:
  std::vector<std::string> var_names_;
  RowToColLookup basic_vars_;
  std::vector<size_t> opposites_;

  void buildOppositesFromNames();
};
}  // namespace bilp
}  // namespace lang
}  // namespace tile
}  // namespace vertexai
