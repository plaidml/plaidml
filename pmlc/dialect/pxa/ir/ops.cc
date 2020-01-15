// Copyright 2019, Intel Corporation

#include <iostream>

#include "pmlc/dialect/pxa/ir/ops.h"

namespace pmlc::dialect::pxa {

#define GET_OP_CLASSES
#include "pmlc/dialect/pxa/ir/ops.cc.inc"

void AffineParallelForOp::build(Builder* builder, OperationState& result, ArrayRef<int64_t> ranges) {
  llvm::SmallVector<mlir::AffineExpr, 8> range_exprs;
  // Make range expressions for each range
  for (int64_t range : ranges) {
    range_exprs.push_back(builder->getAffineConstantExpr(range));
  }
  // Make the maps (handling the 0-dim case carefully)
  if (ranges.size()) {
    result.addAttribute("ranges", AffineMapAttr::get(AffineMap::get(0, 0, range_exprs)));
    result.addAttribute("transform", AffineMapAttr::get(builder->getMultiDimIdentityMap(ranges.size())));
  } else {
    result.addAttribute("ranges", AffineMapAttr::get(AffineMap::get(builder->getContext())));
    result.addAttribute("transform", AffineMapAttr::get(AffineMap::get(builder->getContext())));
  }
  // Create a region and a block for the body.
  auto bodyRegion = result.addRegion();
  auto body = new mlir::Block();
  // Add all the args
  for (size_t i = 0; i < ranges.size(); i++) {
    body->addArgument(IndexType::get(builder->getContext()));
  }
  bodyRegion->push_back(body);
  // Terminate
  ensureTerminator(*bodyRegion, *builder, result.location);
}

}  // namespace pmlc::dialect::pxa
