#include "pmlc/util/math/matrix.h"

#include <algorithm>
#include <set>
#include <string>
#include <tuple>
#include <utility>

#include "llvm/Support/FormatVariadic.h"

namespace pmlc::util::math {

struct DualMatrix {
  size_t size_;
  Matrix lhs_;
  Matrix rhs_;

  explicit DualMatrix(const Matrix& m) : size_(m.size1()), lhs_(IdentityMatrix(size_)), rhs_(m) {}

  void swapRows(size_t r1, size_t r2) {
    lhs_.swapRows(r1, r2);
    rhs_.swapRows(r1, r2);
  }
  void multRow(size_t r, Rational v) {
    lhs_.multRow(r, v);
    rhs_.multRow(r, v);
  }
  void addMultRow(size_t d, size_t s, Rational v) {
    lhs_.addRowMultToRow(d, s, v);
    rhs_.addRowMultToRow(d, s, v);
  }

  std::string toString() const {
    std::string r = "";
    for (size_t i = 0; i < size_; i++) {
      for (size_t j = 0; j < size_; j++) {
        r += llvm::formatv("{0,4}, ", to_string(lhs_(i, j)));
      }
      r += "      ";
      for (size_t j = 0; j < size_; j++) {
        r += llvm::formatv("{0,4}, ", to_string(rhs_(i, j)));
      }
      r += "\n";
    }
    return r;
  }

  // Do elimination, returns false if singular
  bool invert() {
    // First zero the lower triangle of the RHS
    for (size_t i = 0; i < size_; i++) {
      // Pivot first to non-zero entry in diagonal
      bool found_nonzero = false;
      for (size_t j = i; j < size_; j++) {
        if (rhs_(j, i) != 0) {
          found_nonzero = true;
          swapRows(i, j);
          break;
        }
      }
      if (!found_nonzero) {
        return false;
      }
      // Divide row to make (i, i) == 1
      multRow(i, 1 / rhs_(i, i));
      // Zero this column all the way down
      for (size_t j = i + 1; j < size_; j++) {
        addMultRow(j, i, -rhs_(j, i));
      }
    }
    // Now, zero the upper triangle of the RHS
    for (int i = size_ - 1; i >= 0; i--) {
      // Zero this column all the way up
      for (int j = i - 1; j >= 0; j--) {
        addMultRow(j, i, -rhs_(j, i));
      }
    }
    return true;
  }
};

void Matrix::swapRows(size_t r, size_t s) {
  for (size_t i = 0; i < size2(); i++) {
    std::swap((*this)(r, i), (*this)(s, i));
  }
}

void Matrix::multRow(size_t r, Rational multiplier) {
  for (size_t i = 0; i < size2(); i++) {
    (*this)(r, i) *= multiplier;
  }
}

void Matrix::addRowMultToRow(size_t dest_row, size_t src_row, const Rational& multiplier) {
  if (multiplier != 0) {
    for (size_t i = 0; i < size2(); i++) {
      (*this)(dest_row, i) += multiplier * (*this)(src_row, i);
    }
  }
}

void Matrix::makePivotAt(size_t row, size_t col) {
  if ((*this)(row, col) == 0) {
    throw std::runtime_error("Cannot pivot matrix at entry containing 0");
  }
  for (size_t r = 0; r < size1(); ++r) {
    if (r == row) {
      continue;
    }
    addRowMultToRow(r, row, -(*this)(r, col) / (*this)(row, col));
  }
  multRow(row, 1 / (*this)(row, col));
}

bool Matrix::invert() {
  if (size1() != size2()) {
    throw std::runtime_error("Trying to invert non-square matrix");
  }
  DualMatrix dm(*this);
  if (!dm.invert()) {
    return false;
  }
  *this = dm.lhs_;
  return true;
}

std::string Matrix::toString() const {
  std::string ret;
  ret += "\n";
  for (size_t i = 0; i < size1(); ++i) {
    ret += "[ ";
    for (size_t j = 0; j < size2(); ++j) {
      ret += ((*this)(i, j)).str() + "\t";
    }
    ret += "]\n";
  }
  return ret;
}

bool Matrix::operator==(const Matrix& m) {
  if (size1() != m.size1()) {
    return false;
  }
  if (size2() != m.size2()) {
    return false;
  }
  for (size_t i = 0; i < size1(); i++) {
    for (size_t j = 0; j < size2(); j++) {
      if ((*this)(i, j) != m(i, j)) {
        return false;
      }
    }
  }
  return true;
}

Vector VectorLit(const std::vector<Rational>& vec) {
  Vector r(vec.size());
  for (size_t i = 0; i < vec.size(); i++) {
    r(i) = vec[i];
  }
  return r;
}

Matrix MatrixLit(const std::vector<std::vector<Rational>>& vecs) {
  size_t rows = vecs.size();
  size_t columns = vecs[0].size();
  Matrix r(rows, columns);
  for (size_t i = 0; i < rows; i++) {
    if (vecs[i].size() != columns) {
      throw std::runtime_error("Non-rectangular matrix literal");
    }
    for (size_t j = 0; j < columns; j++) {
      r(i, j) = vecs[i][j];
    }
  }
  return r;
}

bool operator==(const Vector& a, const Vector& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); i++) {
    if (a(i) != b(i)) {
      return false;
    }
  }
  return true;
}

std::tuple<Matrix, Vector> FromPolynomials(const std::vector<Polynomial<Rational>>& polys) {
  std::set<std::string> vars;
  for (size_t i = 0; i < polys.size(); i++) {
    for (const auto& kvp : polys[i].getMap()) {
      if (kvp.first != "") {
        vars.insert(kvp.first);
      }
    }
  }
  Matrix mat(polys.size(), vars.size());
  Vector vec(polys.size());
  for (size_t i = 0; i < polys.size(); i++) {
    vec(i) = polys[i].constant();
    size_t j = 0;
    for (const auto& v : vars) {
      mat(i, j) = polys[i][v];
      j++;
    }
  }
  return std::tie(mat, vec);
}

struct HermiteCompute {
  size_t rows_;
  size_t columns_;
  Matrix lhs_;

  void swap(size_t i, size_t j) { lhs_.swapRows(i, j); }

  void mult(size_t i, Integer m) {
    if (m != 1 && m != -1) {
      throw std::runtime_error("Cannot multiply row by nonunit constant in computing HNF.");
    }
    lhs_.multRow(i, m);
  }

  void addMult(size_t d, size_t s, Integer m) {
    IVLOG(6, "  Adding " << m << " * row " << s << " to row " << d);
    lhs_.addRowMultToRow(d, s, m);
  }

  void eliminate(size_t i, size_t j) {
    IVLOG(5, "    Eliminate " << i << ", " << j);
    if (lhs_(j, i) == 0) {
      IVLOG(5, "      Already 0, nothing to do");
      return;
    }
    Integer x, y;
    IVLOG(5, "      Computing XGCD of " << lhs_(i, i) << " and " << lhs_(j, i));
    Rational o = XGCD(lhs_(i, i), lhs_(j, i), x, y);
    IVLOG(5, "o = " << o << ", x = " << x << ", y = " << y);
    if (Abs(o) != lhs_(i, i)) {
      if (Abs(o) == Abs(lhs_(j, i))) {
        IVLOG(5, "      Swapping entry");
        swap(i, j);
        if (lhs_(i, i) < 0) {
          mult(i, -1);
        }
      } else {
        IVLOG(5, "      Updating entry");
        euclidean_reduce(i, j, i);
      }
    }
    Rational m = numerator(-lhs_(j, i) / o);
    IVLOG(5, "  m = " << m);
    addMult(j, i, numerator(-lhs_(j, i) / o));
  }

  void euclidean_reduce(size_t i, size_t j, size_t col) {
    Rational a = lhs_(i, col);
    Rational b = lhs_(j, col);
    if (a < 0) {
      a = -a;
      IVLOG(5, "    Negating row " << i)
      mult(i, -1);
      IVLOG(6, "  state\n" << toString());
    }
    if (b < 0) {
      b = -b;
      IVLOG(5, "    Negating row " << j)
      mult(j, -1);
      IVLOG(6, "  state\n" << toString());
    }
    if (a < b) {
      swap(i, j);
      IVLOG(6, "  state\n" << toString());
      a = lhs_(i, col);
      b = lhs_(j, col);
    }

    // Main Euclidean algorithm
    Rational r;
    Integer q = RatDiv(a, b, r);
    IVLOG(6, "Quotient " << q << ", Remainder " << r);
    while (true) {
      addMult(i, j, -q);
      swap(i, j);
      IVLOG(6, "  a = " << a << ", b = " << b << ", state\n" << toString());
      if (r == 0) {
        IVLOG(6, "Remainder 0, stopping");
        break;
      }
      a = b;
      b = r;
      q = RatDiv(a, b, r);
      IVLOG(6, "Quotient " << q << ", Remainder " << r);
    }
  }

  void normalize(size_t i, size_t j) {
    Integer m = -Floor(lhs_(j, i) / lhs_(i, i));
    addMult(j, i, m);
  }

  std::string toString() {
    std::stringstream ss;
    for (size_t i = 0; i < rows_; i++) {
      for (size_t j = 0; j < columns_; j++) {
        ss << lhs_(i, j).str() << " ";
      }
    }
    return ss.str();
  }

 public:
  explicit HermiteCompute(const Matrix& m) : rows_(m.size1()), columns_(m.size2()), lhs_(m) {}

  bool compute() {
    if (rows_ < columns_) {
      return false;
    }
    IVLOG(4, "Computing HNF, initial state\n" << toString());
    // TODO(T132): Technically, this ordering of the algorithm may be
    // exponential in the bit representation of the rational numbers
    // There is a polynomial time version, but the pivoting is tricker
    // so I'm skipping it for now.
    for (size_t i = 0; i < columns_; i++) {
      IVLOG(5, "Fixing column " << i);
      IVLOG(5, "  state\n" << toString());
      // First, we need to position (i, i) with a non-zero entry if possible
      for (size_t j = i; j < rows_; j++) {
        if (lhs_(j, i) != 0) {
          IVLOG(5, "  Swapping " << i << " and " << j);
          swap(i, j);
          IVLOG(6, "  state\n" << toString());
          break;
        }
      }
      // If they are all zeros, whatevs, on to the next column
      if (lhs_(i, i) == 0) {
        IVLOG(5, "  Skipping due to zeros");
        continue;
      }
      // Otherwise, fix sign and combine with all rows below
      if (lhs_(i, i) < 0) {
        IVLOG(6, " Multiplying " << i << " by -1");
        mult(i, -1);
        IVLOG(6, "  state\n" << toString());
      }
      for (size_t j = i + 1; j < rows_; j++) {
        eliminate(i, j);
        IVLOG(6, "  state\n" << toString());
      }
      // And normalize all rows above
      for (size_t j = 0; j < i; j++) {
        normalize(i, j);
        IVLOG(6, "  state\n" << toString());
      }
    }
    IVLOG(4, "Final state\n" << toString());
    return true;
  }
};

bool HermiteNormalForm(Matrix& m) {  // NOLINT(runtime/references)
  HermiteCompute hc(m);
  bool r = hc.compute();
  m = hc.lhs_;
  return r;
}

}  // namespace pmlc::util::math
