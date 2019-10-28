// Copyright 2019, Intel Corporation

#include "pmlc/dialect/stripe/util.h"

#include "mlir/IR/Builders.h"

#include "pmlc/dialect/stripe/dialect.h"
#include "pmlc/dialect/stripe/ops.h"

using mlir::NamedAttribute;
using mlir::OpBuilder;
using pmlc::dialect::stripe::ParallelForOp;
using pmlc::dialect::stripe::TerminateOp;

namespace pmlc {
namespace dialect {
namespace stripe {

void createMainParallelFor(mlir::FuncOp funcOp) {
  auto& region = funcOp.getBody();
  OpBuilder builder(region);
  auto src = &region.front();
  auto it = src->begin();
  auto forOp = builder.create<ParallelForOp>(funcOp.getLoc(), builder.getI64ArrayAttr({}));
  auto attrs = llvm::SmallVector<NamedAttribute, 1>{
      {builder.getIdentifier("main"), builder.getUnitAttr()},
  };
  forOp.setAttr(dialect::stripe::Dialect::getStripeAttrsName(), builder.getDictionaryAttr(attrs));
  forOp.setAttr("name", builder.getStringAttr("main"));
  auto block = builder.createBlock(&forOp.inner());
  auto& dst = block->getOperations();
  dst.splice(dst.end(), src->getOperations(), it, src->end());

  builder.setInsertionPointToEnd(src);
  builder.create<TerminateOp>(funcOp.getLoc());
}

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
