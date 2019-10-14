// Copyright 2019, Intel Corporation

#include "pmlc/dialect/stripe/util.h"

#include "pmlc/dialect/stripe/dialect.h"
#include "pmlc/dialect/stripe/ops.h"

#include "mlir/IR/Builders.h"
#include "mlir/TableGen/Argument.h"

using mlir::NamedAttribute;
using mlir::OpBuilder;
using mlir::Value;
using pmlc::dialect::stripe::AffineConstOp;
using pmlc::dialect::stripe::AffineType;
using pmlc::dialect::stripe::ParallelForOp;
using pmlc::dialect::stripe::RefineOp;
using pmlc::dialect::stripe::TensorRefType;
using pmlc::dialect::stripe::TerminateOp;

namespace pmlc {
namespace dialect {
namespace stripe {

void createMainParallelFor(mlir::FuncOp funcOp) {
  auto& region = funcOp.getBody();
  OpBuilder builder(region);
  auto body = region.begin();
  auto it = body->begin();
  auto forOp = builder.create<ParallelForOp>(funcOp.getLoc(), builder.getI64ArrayAttr({}));
  auto attrs = llvm::SmallVector<NamedAttribute, 1>{
      {builder.getIdentifier("main"), builder.getUnitAttr()},
  };
  forOp.setAttr(dialect::stripe::Dialect::getStripeAttrsName(), builder.getDictionaryAttr(attrs));
  forOp.setAttr("name", builder.getStringAttr("main"));
  auto block = builder.createBlock(&forOp.inner());
  block->getOperations().splice(block->getOperations().end(), body->getOperations(), it, body->end());

  // Inject RefineOp between each block argument and first usage
  builder.setInsertionPointToStart(&region.front());
  auto constOp = builder.create<AffineConstOp>(  //
      funcOp.getLoc(),                           //
      builder.getType<AffineType>(),             //
      builder.getI64IntegerAttr(0));
  auto zero = constOp.result();
  for (auto arg : funcOp.getArguments()) {
    auto tensorRefType = arg->getType().cast<TensorRefType>();
    llvm::SmallVector<Value*, 4> offsets(tensorRefType.getRank(), zero);
    auto refineOp = builder.create<RefineOp>(funcOp.getLoc(), tensorRefType, arg, offsets);
    arg->replaceAllUsesWith(refineOp);
    refineOp.setOperand(0, arg);
  }
  builder.setInsertionPointToEnd(&region.front());
  builder.create<TerminateOp>(funcOp.getLoc());
}

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
