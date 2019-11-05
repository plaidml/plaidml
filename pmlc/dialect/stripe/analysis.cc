// Copyright 2019, Intel Corporation

#include "pmlc/dialect/stripe/analysis.h"

#include <utility>

#include <boost/math/common_factor.hpp>

#include "pmlc/dialect/stripe/dialect.h"

namespace pmlc {
namespace dialect {
namespace stripe {

AffineRange::AffineRange(int64_t _min, int64_t _max, uint64_t _stride) : min(_min), max(_max), stride(_stride) {
  if (min == max) {
    stride = 0;
  }
}

AffineRange& AffineRange::operator*=(int64_t x) {
  min *= x;
  max *= x;
  if (x < 0) {
    std::swap(min, max);
  }
  stride *= x;
  return *this;
}

AffineRange& AffineRange::operator+=(const AffineRange& x) {
  min += x.min;
  max += x.max;
  stride = boost::math::gcd(stride, x.stride);
  return *this;
}

AffineRange& AffineRange::operator|=(const AffineRange& x) {
  min = std::min(min, x.min);
  max = std::max(max, x.max);
  stride = boost::math::gcd(stride, x.stride);
  return *this;
}

AffineRange::AffineRange(const AffinePolynomial& poly) : min(poly.constant), max(poly.constant), stride(0) {
  // Go over each term and adjust
  for (const auto& kvp : poly.terms) {
    // Update the stride
    stride = boost::math::gcd(stride, uint64_t(std::abs(kvp.second)));
    // Extract the parallel for this affine is an argument of
    auto pf = mlir::cast<ParallelForOp>(kvp.first->getOwner()->getParentOp());
    // Extract the appropriate attribute from ranges
    Attribute ra = pf.ranges().getValue()[kvp.first->getArgNumber()];
    // Turn the range into an integer
    int64_t range = ra.cast<IntegerAttr>().getInt();
    // Update min/max
    if (kvp.second >= 0) {
      max += (range - 1) * kvp.second;
    } else {
      min += (range - 1) * kvp.second;
    }
  }
}

bool FlatTensorAccess::operator<(const FlatTensorAccess& rhs) const {
  if (base != rhs.base) {
    return base < rhs.base;
  }
  return access < rhs.access;
}

bool FlatTensorAccess::operator==(const FlatTensorAccess& rhs) const {
  return base == rhs.base && access == rhs.access;
}

FlatTensorAccess ComputeAccess(Value* tensor) {
  FlatTensorAccess ret;
  if (auto bop = tensor->getDefiningOp()) {
    if (auto op = mlir::dyn_cast<AllocateOp>(bop)) {
      ret.base = op.result();
      ret.base_type = op.layout();
      ret.access.resize(ret.base_type.getRank());
    } else if (auto op = mlir::dyn_cast<RefineOp>(bop)) {
      ret = ComputeAccess(op.in());
      for (size_t i = 0; i < ret.access.size(); i++) {
        ret.access[i] += AffinePolynomial(op.getOffset(i));
      }
    } else {
      throw std::runtime_error("Invalid tensor value in ComputeAccess");
    }
  } else if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(tensor)) {
    auto parentOp = arg->getOwner()->getParentOp();
    auto funcOp = mlir::dyn_cast<mlir::FuncOp>(parentOp);
    if (!funcOp) {
      throw std::runtime_error("Invalid tensor value: block argument not contained by FuncOp");
    }
    auto attrName = stripe::Dialect::getDialectAttrName("layout");
    auto attr = funcOp.getArgAttrOfType<mlir::TypeAttr>(arg->getArgNumber(), attrName);
    assert(attr && "Expected 'layout' attribute in TensorRefType function argument");
    ret.base = tensor;
    ret.base_type = attr.getValue().cast<TensorType>();
    ret.access.resize(ret.base_type.getRank());
  } else {
    throw std::runtime_error("Invalid tensor value");
  }
  return ret;
}

bool SafeConstraintInterior(ParallelForOp op) {
  // Get an iterator to the begining of the interior
  auto block = &op.inner().front();
  // Get the penultimate Op (ignoring the terminator), which should be a constraint
  auto it_con = std::prev(block->end(), 2);
  // Check that it's good
  if (it_con == block->end() || !mlir::isa<ConstraintOp>(*it_con)) {
    return false;
  }
  // Check that all prior ops are no-side-effect and fail if not
  for (auto it = block->begin(); it != it_con; ++it) {
    if (!it->hasNoSideEffect()) {
      return false;
    }
  }
  return true;
}

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
