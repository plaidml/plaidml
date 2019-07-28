// Copyright 2019, Intel Corporation

#include "pmlc/dialect/scalar/ops.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"

#define DEBUG_TYPE "pml_scalar"

namespace pmlc {
namespace dialect {
namespace scalar {

namespace {

using UnaryCalculate = std::function<double(double)>;
using BinaryCalculate = std::function<double(double, double)>;

Attribute constFoldUnaryOp(ArrayRef<Attribute> operands, UnaryCalculate calculate) {
  assert(operands.size() == 1 && "unary op takes one operand");
  if (auto op = operands[0].dyn_cast_or_null<FloatAttr>()) {
    return FloatAttr::get(op.getType(), calculate(op.getValueAsDouble()));
  }
  if (auto op = operands[0].dyn_cast_or_null<IntegerAttr>()) {
    return IntegerAttr::get(op.getType(), calculate(op.getInt()));
  }
  return {};
}

Attribute constFoldBinaryOp(ArrayRef<Attribute> operands, BinaryCalculate calculate) {
  assert(operands.size() == 2 && "binary op takes two operands");
  if (auto lhs = operands[0].dyn_cast_or_null<FloatAttr>()) {
    if (auto rhs = operands[1].dyn_cast_or_null<FloatAttr>()) {
      return FloatAttr::get(lhs.getType(), calculate(lhs.getValueAsDouble(), rhs.getValueAsDouble()));
    }
    if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>()) {
      return FloatAttr::get(lhs.getType(), calculate(lhs.getValueAsDouble(), rhs.getInt()));
    }
  } else if (auto lhs = operands[0].dyn_cast_or_null<IntegerAttr>()) {
    if (auto rhs = operands[1].dyn_cast_or_null<FloatAttr>()) {
      return FloatAttr::get(rhs.getType(), calculate(lhs.getInt(), rhs.getValueAsDouble()));
    }
    if (auto rhs = operands[1].dyn_cast_or_null<IntegerAttr>()) {
      return IntegerAttr::get(lhs.getType(), calculate(lhs.getInt(), rhs.getInt()));
    }
  }
  return {};
}

/// The matcher that matches a constant scalar value.
struct ConstantValueMatcher {
  double value;

  bool match(Operation* op) {
    Attribute attr;
    if (!mlir::detail::constant_op_binder<Attribute>(&attr).match(op)) {
      return false;
    }
    auto type = op->getResult(0)->getType();
    if (type.isa<ScalarType>()) {
      if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
        return intAttr.getValue() == value;
      }
      if (auto floatAttr = attr.dyn_cast<FloatAttr>()) {
        return floatAttr.getValueAsDouble() == value;
      }
    }
    return false;
  }
};

/// Matches a constant scalar zero.
inline ConstantValueMatcher m_Zero() { return ConstantValueMatcher{0}; }

/// Matches a constant scalar one.
inline ConstantValueMatcher m_One() { return ConstantValueMatcher{1}; }

}  // namespace

OpFoldResult ScalarConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return getValue();
}

OpFoldResult AddOp::fold(ArrayRef<Attribute> operands) {
  /// add(x, 0) -> x
  if (matchPattern(rhs(), m_Zero())) {
    return lhs();
  }
  return constFoldBinaryOp(operands, [](double a, double b) { return a + b; });
}

OpFoldResult DivOp::fold(ArrayRef<Attribute> operands) {
  // Don't fold if it requires division by zero.
  if (matchPattern(rhs(), m_Zero())) {
    return {};
  }
  /// div(x, 1) -> x
  if (matchPattern(rhs(), m_One())) {
    return lhs();
  }
  /// div(0, x) -> 0
  if (matchPattern(lhs(), m_Zero())) {
    return Builder(getContext()).getZeroAttr(getType());
  }
  return constFoldBinaryOp(operands, [](double a, double b) { return a / b; });
}

OpFoldResult MulOp::fold(ArrayRef<Attribute> operands) {
  /// mul(x, 0) -> 0
  if (matchPattern(rhs(), m_Zero())) {
    return rhs();
  }
  /// mul(x, 1) -> x
  if (matchPattern(rhs(), m_One())) {
    return lhs();
  }
  return constFoldBinaryOp(operands, [](double a, double b) { return a * b; });
}

OpFoldResult SubOp::fold(ArrayRef<Attribute> operands) {
  // sub(x, x) -> 0
  if (lhs() == rhs()) {
    return Builder(getContext()).getZeroAttr(getType());
  }
  /// sub(x, 0) -> x
  if (matchPattern(rhs(), m_Zero())) {
    return lhs();
  }
  return constFoldBinaryOp(operands, [](double a, double b) { return a - b; });
}

OpFoldResult NegOp::fold(ArrayRef<Attribute> operands) {
  return constFoldUnaryOp(operands, [](double x) { return -x; });
}

#define GET_OP_CLASSES
#include "pmlc/dialect/scalar/ops.cpp.inc"

}  // namespace scalar
}  // namespace dialect
}  // namespace pmlc
