// Copyright 2020 Intel Corporation

#include "mlir/Dialect/StandardOps/IR/Ops.h"

#include "pmlc/dialect/abi/transforms/pass_detail.h"

namespace pmlc::dialect::abi {
namespace {

class DenormalizeConstantsPass final
    : public DenormalizeConstantsPassBase<DenormalizeConstantsPass> {
public:
  void runOnOperation() final;
};

void DenormalizeConstantsPass::runOnOperation() {
  auto loopOp = getOperation();
  auto networkOp = loopOp.getInitTerminator();
  auto networkFieldTypes = loopOp.getAttrOfType<mlir::ArrayAttr>(
      abi::LoopOp::getNetworkFieldTypesAttrName());
  mlir::SmallVector<mlir::Attribute, 8> newNetworkFieldTypes;
  mlir::OpBuilder builder{loopOp.bodyRegion()};
  unsigned idx = 0;
  for (auto tyAttr : networkFieldTypes) {
    auto val = networkOp.getOperand(idx);
    if (auto constOp = val.getDefiningOp<mlir::ConstantOp>()) {
      auto denormOp = constOp.clone();
      builder.insert(denormOp);
      auto arg = loopOp.bodyEntryBlock()->getArgument(idx);
      arg.replaceAllUsesWith(denormOp);
      loopOp.bodyEntryBlock()->eraseArgument(idx);
      networkOp.getOperation()->eraseOperand(idx);
      if (constOp.use_empty()) {
        constOp.erase();
      }
    } else {
      newNetworkFieldTypes.emplace_back(tyAttr);
      ++idx;
    }
  }
  if (networkFieldTypes.size() != newNetworkFieldTypes.size()) {
    loopOp.setAttr(abi::LoopOp::getNetworkFieldTypesAttrName(),
                   builder.getArrayAttr(newNetworkFieldTypes));
  }
}

} // namespace

std::unique_ptr<mlir::Pass> createDenormalizeConstantsPass() {
  return std::make_unique<DenormalizeConstantsPass>();
}

} // namespace pmlc::dialect::abi
