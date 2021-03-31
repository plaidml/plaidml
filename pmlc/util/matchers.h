// Copyright 2020 Intel Corporation

#pragma once

#include <tuple>

#include "mlir/IR/Matchers.h"

// TODO: move to upstream
namespace mlir {

namespace detail {

template <typename OpType, typename... OperandMatchers>
struct PartialRecursivePatternMatcher {
  explicit PartialRecursivePatternMatcher(OperandMatchers... matchers)
      : operandMatchers(matchers...) {}
  bool match(Operation *op) {
    if (!isa<OpType>(op) || op->getNumOperands() < sizeof...(OperandMatchers)) {
      return false;
    }
    bool res = true;
    enumerate(operandMatchers, [&](size_t index, auto &matcher) {
      res &= matchOperandOrValueAtIndex(op, index, matcher);
    });
    return res;
  }
  std::tuple<OperandMatchers...> operandMatchers;
};

struct CaptureMatcher {
  explicit CaptureMatcher(mlir::Value *capture) : capture(capture) {}
  bool match(mlir::Value value) {
    *capture = value;
    return true;
  }
  mlir::Value *capture;
};

template <typename Pattern>
struct CaptureAndContinueMatcher {
  CaptureAndContinueMatcher(mlir::Value *capture, Pattern pattern)
      : capture(capture), pattern(pattern) {}
  bool match(mlir::Value value) {
    *capture = value;
    return matchPattern(value, pattern);
  }
  mlir::Value *capture;
  Pattern pattern;
};

} // namespace detail

template <typename OpType, typename... Matchers>
auto m_PartialOp(Matchers... matchers) {
  return detail::PartialRecursivePatternMatcher<OpType, Matchers...>(
      matchers...);
}

inline auto m_Capture(mlir::Value *value) {
  return detail::CaptureMatcher(value);
}

template <typename Pattern>
inline auto m_Capture(mlir::Value *value, Pattern pattern) {
  return detail::CaptureAndContinueMatcher<Pattern>(value, pattern);
}

} // namespace mlir
