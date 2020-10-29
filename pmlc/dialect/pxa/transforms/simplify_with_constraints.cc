// Copyright 2020, Intel Corporation
#include "pmlc/dialect/pxa/transforms/simplify_with_constraints.h"

#include <memory>

#include "pmlc/dialect/pxa/analysis/affine_constraints.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/pass_detail.h"
#include "pmlc/dialect/pxa/transforms/passes.h"
#include "llvm/ADT/TypeSwitch.h"

namespace pmlc::dialect::pxa {

namespace {
class SimplifyWithConstraintsPass final
    : public SimplifyWithConstraintsBase<SimplifyWithConstraintsPass> {
public:
  void runOnFunction() {
    mlir::FuncOp func = getFunction();
    func.walk(simplifyAffineMapsWithConstraints);
  }
};

template <typename OpTy>
void simplifyLoadMap(OpTy op) {
  mlir::AffineMap affineMap = op.getAffineMap();
  mlir::Operation::operand_range mapOperands = op.getMapOperands();
  mlir::AffineValueMap oldMap(affineMap, mapOperands);
  mlir::AffineValueMap newMap = simplifyMapWithConstraints(oldMap);
  if (newMap.getAffineMap() == oldMap.getAffineMap())
    return;
  auto newMapAttr = mlir::AffineMapAttr::get(newMap.getAffineMap());
  op.setAttr(op.getMapAttrName(), newMapAttr);
  op.indicesMutable().assign(newMap.getOperands());
}

template <typename OpTy>
void simplifyReduceMap(OpTy op) {
  mlir::AffineMap affineMap = op.getAffineMap();
  mlir::Operation::operand_range mapOperands = op.getMapOperands();
  mlir::AffineValueMap oldMap(affineMap, mapOperands);
  mlir::AffineValueMap newMap = simplifyMapWithConstraints(oldMap);
  if (newMap.getAffineMap() == oldMap.getAffineMap())
    return;
  auto newMapAttr = mlir::AffineMapAttr::get(newMap.getAffineMap());
  op.setAttr(op.getMapAttrName(), newMapAttr);
  op.idxsMutable().assign(newMap.getOperands());
}

} // namespace

void simplifyAffineMapsWithConstraints(mlir::Operation *op) {
  mlir::TypeSwitch<mlir::Operation *>(op)
      .Case<PxaLoadOp>(simplifyLoadMap<PxaLoadOp>)
      .Case<PxaVectorLoadOp>(simplifyLoadMap<PxaVectorLoadOp>)
      .Case<PxaReduceOp>(simplifyReduceMap<PxaReduceOp>)
      .Case<PxaVectorReduceOp>(simplifyReduceMap<PxaVectorReduceOp>);
}

std::unique_ptr<mlir::Pass> createSimplifyWithConstraintsPass() {
  return std::make_unique<SimplifyWithConstraintsPass>();
}

} // namespace pmlc::dialect::pxa
