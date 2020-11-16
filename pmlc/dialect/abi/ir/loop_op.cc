// Copyright 2020, Intel Corporation

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"

#include "pmlc/dialect/abi/ir/dialect.h"

namespace pmlc::dialect::abi {

void LoopOp::build(::mlir::OpBuilder &odsBuilder,
                   ::mlir::OperationState &odsState) {
  odsState.addRegion();
  odsState.addRegion();
  odsState.addRegion();
}

mlir::Block *LoopOp::getBodyEntryBlock() { return &bodyRegion().front(); }
mlir::Block *LoopOp::getFiniEntryBlock() { return &finiRegion().front(); }

YieldOp LoopOp::getInitTerminator() {
  return mlir::cast<YieldOp>(initRegion().back().getTerminator());
}

TerminatorOp LoopOp::getFiniTerminator() {
  return mlir::cast<TerminatorOp>(finiRegion().back().getTerminator());
}

} // namespace pmlc::dialect::abi
