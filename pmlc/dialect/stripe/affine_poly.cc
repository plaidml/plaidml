// Copyright 2019, Intel Corporation

#include "pmlc/dialect/stripe/affine_poly.h"

#include <utility>

#include <boost/math/common_factor.hpp>

#include "pmlc/dialect/stripe/dialect.h"
#include "pmlc/dialect/stripe/ops.h"

namespace pmlc {
namespace dialect {
namespace stripe {

AffinePolynomial::AffinePolynomial() : constant(0) {}

AffinePolynomial::AffinePolynomial(int64_t x) : constant(x) {}

AffinePolynomial::AffinePolynomial(Value* value) : constant(0) {
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    // This is a basic index: done
    terms.emplace(arg, 1);
    return;
  }
  auto defOp = value->getDefiningOp();
  if (auto op = mlir::dyn_cast<AffinePolyOp>(defOp)) {
    constant = op.offset().getSExtValue();
    for (size_t i = 0; i < op.coeffs().size(); i++) {
      *this += AffinePolynomial(op.getOperand(i)) * op.getCoeff(i);
    }
  } else {
    throw std::runtime_error("Invalid affine in ComputeAffineRange");
  }
}

AffinePolynomial& AffinePolynomial::operator*=(int64_t x) {
  constant *= x;
  if (x == 0) {
    terms.clear();
  } else {
    for (auto& kvp : terms) {
      kvp.second *= x;
    }
  }
  return *this;
}

AffinePolynomial& AffinePolynomial::operator+=(const AffinePolynomial& x) {
  constant += x.constant;
  for (const auto& kvp : x.terms) {
    terms[kvp.first] += kvp.second;
    if (terms[kvp.first] == 0) {
      terms.erase(kvp.first);
    }
  }
  return *this;
}

bool AffinePolynomial::operator<(const AffinePolynomial& rhs) const {
  if (constant < rhs.constant) {
    return true;
  }
  if (constant > rhs.constant) {
    return false;
  }
  return terms < rhs.terms;
}

bool AffinePolynomial::operator==(const AffinePolynomial& rhs) const {
  return constant == rhs.constant && terms == rhs.terms;
}

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
