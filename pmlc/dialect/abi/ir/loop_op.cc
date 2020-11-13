// Copyright 2020, Intel Corporation

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"

#include "pmlc/dialect/abi/ir/dialect.h"

namespace pmlc::dialect::abi {

void LoopOp::build(::mlir::OpBuilder &odsBuilder,
                   ::mlir::OperationState &odsState) {
  odsState.addAttribute("networkFieldTypes", odsBuilder.getTypeArrayAttr({}));
  odsState.addRegion();
  odsState.addRegion();
  odsState.addRegion();
}

std::vector<mlir::Type> LoopOp::getNetworkFieldTypes() {
  std::vector<mlir::Type> result;
  auto arrayAttr = networkFieldTypes();
  if (arrayAttr) {
    for (auto attr : arrayAttr) {
      if (auto tyAttr = attr.dyn_cast<mlir::TypeAttr>()) {
        result.emplace_back(tyAttr.getValue());
      }
    }
  }
  return result;
}

unsigned LoopOp::getNumNetworkFields() {
  auto arrayAttr = networkFieldTypes();
  if (arrayAttr) {
    return arrayAttr.size();
  }
  return 0;
}

void LoopOp::setNetworkFieldTypes(mlir::TypeRange types) {
  mlir::SmallVector<mlir::Attribute, 8> attrs;
  for (auto ty : types) {
    attrs.emplace_back(mlir::TypeAttr::get(ty));
  }
  auto arrayAttr = mlir::ArrayAttr::get(attrs, getContext());
  networkFieldTypesAttr(arrayAttr);
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
