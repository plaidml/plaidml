// Copyright 2020, Intel Corporation

#include <string>

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/SPIRV/Serialization.h"
#include "mlir/Transforms/DialectConversion.h"

#include "pmlc/conversion/comp_to_llvm/pass_detail.h"
#include "pmlc/conversion/comp_to_llvm/passes.h"
#include "pmlc/conversion/comp_to_llvm/utils.h"
#include "pmlc/dialect/comp/ir/dialect.h"

namespace pmlc::conversion::comp_to_llvm {

using namespace mlir; // NOLINT

namespace comp = pmlc::dialect::comp;

static constexpr const char *kSpirvBinPrefix = "_pmlc_spirv_bin_";
static constexpr const char *kSpirvKernelPrefix = "_pmlc_spirv_kernel_";

LogicalResult serializeSpirvKernels(ModuleOp op, BinaryModulesMap &map) {
  OpBuilder builder(op.getBodyRegion());
  std::vector<Operation *> toErase;

  WalkResult serializeWalk =
      op.walk([&](spirv::ModuleOp moduleOp) -> WalkResult {
        auto gpuModule =
            dyn_cast<gpu::GPUModuleOp>(moduleOp.getOperation()->getNextNode());
        if (!gpuModule)
          return WalkResult::interrupt();
        std::string gpuModuleName = gpuModule.getName().str();
        std::string binaryName = kSpirvBinPrefix + gpuModuleName;

        // Serialize spirv module.
        SmallVector<uint32_t, 0> moduleBinary;
        if (failed(spirv::serialize(moduleOp, moduleBinary)))
          return WalkResult::interrupt();

        LLVM::GlobalOp binaryOp =
            addGlobalString(builder, moduleOp.getLoc(), binaryName,
                            {reinterpret_cast<char *>(moduleBinary.data()),
                             moduleBinary.size() * 4});
        std::map<std::string, LLVM::GlobalOp> kernelNames;
        gpuModule.walk([&](gpu::GPUFuncOp funcOp) {
          if (!funcOp.isKernel())
            return;
          std::string kernelName = funcOp.getName().str();
          std::string symbol =
              kSpirvKernelPrefix + gpuModuleName + "_" + kernelName;
          // Make into null terminated string.
          auto nullTerminatedName = kernelName;
          nullTerminatedName.push_back('\0');
          LLVM::GlobalOp globalKernelName = addGlobalString(
              builder, funcOp.getLoc(), symbol, nullTerminatedName);
          kernelNames[kernelName] = globalKernelName;
        });
        // Update modules map.
        map[gpuModuleName] = {binaryOp, moduleBinary.size() * 4, kernelNames};

        // Add spirv and gpu modules to erase list.
        toErase.push_back(moduleOp.getOperation());
        toErase.push_back(gpuModule.getOperation());
        return WalkResult::advance();
      });
  if (serializeWalk.wasInterrupted())
    return failure();
  // Finally, erase processed operations.
  for (Operation *opToErase : toErase)
    opToErase->erase();
  return success();
}

} // namespace pmlc::conversion::comp_to_llvm
