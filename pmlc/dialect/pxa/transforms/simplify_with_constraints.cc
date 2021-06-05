// Copyright 2020, Intel Corporation
#include "pmlc/dialect/pxa/transforms/simplify_with_constraints.h"

#include <memory>

#include "pmlc/dialect/pxa/analysis/affine_constraints.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/dialect/pxa/transforms/passes.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir; // NOLINT

namespace pmlc::dialect::pxa {

namespace {
class SimplifyWithConstraintsPass final
    : public SimplifyWithConstraintsBase<SimplifyWithConstraintsPass> {
public:
  void runOnFunction() {
    FuncOp func = getFunction();
    func.walk(simplifyAffineMapsWithConstraints);
  }
};

template <typename OpTy>
void simplifyLoadMap(OpTy op) {
  AffineMap affineMap = op.getAffineMap();
  OperandRange mapOperands = op.getMapOperands();
  AffineValueMap oldMap(affineMap, mapOperands);
  AffineValueMap newMap = simplifyMapWithConstraints(oldMap);
  if (newMap.getAffineMap() == oldMap.getAffineMap())
    return;
  op.mapAttr(AffineMapAttr::get(newMap.getAffineMap()));
  op.idxsMutable().assign(newMap.getOperands());
}

template <typename OpTy>
void simplifyReduceMap(OpTy op) {
  AffineMap affineMap = op.getAffineMap();
  OperandRange mapOperands = op.getMapOperands();
  AffineValueMap oldMap(affineMap, mapOperands);
  AffineValueMap newMap = simplifyMapWithConstraints(oldMap);
  if (newMap.getAffineMap() == oldMap.getAffineMap())
    return;
  op.mapAttr(AffineMapAttr::get(newMap.getAffineMap()));
  op.idxsMutable().assign(newMap.getOperands());
}

} // namespace

void simplifyAffineMapsWithConstraints(Operation *op) {
  TypeSwitch<Operation *>(op)
      .Case<PxaLoadOp>(simplifyLoadMap<PxaLoadOp>)
      .Case<PxaVectorLoadOp>(simplifyLoadMap<PxaVectorLoadOp>)
      .Case<PxaReduceOp>(simplifyReduceMap<PxaReduceOp>)
      .Case<PxaVectorReduceOp>(simplifyReduceMap<PxaVectorReduceOp>);
}

std::unique_ptr<Pass> createSimplifyWithConstraintsPass() {
  return std::make_unique<SimplifyWithConstraintsPass>();
}

} // namespace pmlc::dialect::pxa
