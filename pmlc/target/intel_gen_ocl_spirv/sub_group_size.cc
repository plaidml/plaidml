// Copyright 2020, Intel Corporation

#include "pmlc/target/intel_gen_ocl_spirv/pass_detail.h"
#include "pmlc/target/intel_gen_ocl_spirv/passes.h"

#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"

namespace pmlc::target::intel_gen_ocl_spirv {
namespace spirv = mlir::spirv;

namespace {

class IntelGenOclSetSubgroupSize
    : public IntelGenOclSetSubgroupSizeBase<IntelGenOclSetSubgroupSize> {
public:
  /// Returns entry point function in module or nullptr if there is none.
  spirv::FuncOp getEntryPoint(spirv::ModuleOp module) {
    spirv::FuncOp func = nullptr;
    module.walk([&](spirv::FuncOp op) {
      if (!op.getAttr(spirv::getEntryPointABIAttrName()))
        return mlir::WalkResult::advance();
      func = op;
      return mlir::WalkResult::interrupt();
    });
    return func;
  }
  /// Extracts local size attribute from entry point function.
  mlir::DenseIntElementsAttr getLocalSize(spirv::FuncOp func) {
    auto entryPointAttr = func.getAttrOfType<spirv::EntryPointABIAttr>(
        spirv::getEntryPointABIAttrName());
    if (!entryPointAttr)
      return {};
    return entryPointAttr.local_size();
  }

  void runOnOperation() {
    spirv::ModuleOp module = getOperation();
    spirv::FuncOp func = getEntryPoint(module);
    if (!func)
      return;
    mlir::DenseIntElementsAttr localSize = getLocalSize(func);
    if (!localSize)
      return;
    mlir::APInt localX = *localSize.begin();
    if (localX == 1)
      return;
    // Insert ExecutionMode at the end.
    int32_t localXi32 = static_cast<int32_t>(localX.getZExtValue());
    auto builder = mlir::OpBuilder::atBlockTerminator(&module.getBlock());
    builder.create<spirv::ExecutionModeOp>(func.getLoc(), func,
                                           spirv::ExecutionMode::SubgroupSize,
                                           mlir::ArrayRef<int32_t>{localXi32});
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createSetSubgroupSizePass() {
  return std::make_unique<IntelGenOclSetSubgroupSize>();
}

} // namespace pmlc::target::intel_gen_ocl_spirv
