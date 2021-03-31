// Copyright 2020 Intel Corporation

#pragma once

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Matchers.h"

#include "pmlc/dialect/pxa/ir/ops.h"

namespace pmlc::dialect::pxa {

template <typename A, typename B>
struct PxaReduceOpMatcher {
  PxaReduceOpMatcher(mlir::AtomicRMWKind agg, A aMatcher, B bMatcher)
      : agg(agg), aMatcher(aMatcher), bMatcher(bMatcher) {}
  bool match(mlir::Operation *op) {
    using mlir::detail::matchOperandOrValueAtIndex;
    auto reduce = mlir::dyn_cast<pmlc::dialect::pxa::PxaReduceOp>(op);
    return (reduce && reduce.agg() == agg &&
            matchOperandOrValueAtIndex(op, 0, aMatcher) &&
            matchOperandOrValueAtIndex(op, 1, bMatcher));
  }
  mlir::AtomicRMWKind agg;
  A aMatcher;
  B bMatcher;
};

template <typename A, typename B>
inline auto m_PxaReduceOp(mlir::AtomicRMWKind agg, A aMatcher, B bMatcher) {
  return PxaReduceOpMatcher<A, B>(agg, aMatcher, bMatcher);
}

} // namespace pmlc::dialect::pxa
