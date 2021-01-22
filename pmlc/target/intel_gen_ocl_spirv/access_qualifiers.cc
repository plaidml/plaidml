// Copyright 2020, Intel Corporation

#include "pmlc/target/intel_gen_ocl_spirv/pass_detail.h"
#include "pmlc/target/intel_gen_ocl_spirv/passes.h"

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"

#include "mlir/Support/DebugStringHelper.h"
#include "pmlc/util/logging.h"

namespace pmlc::target::intel_gen_ocl_spirv {
namespace spirv = mlir::spirv;

namespace {

/// This pass adds read only access qualifier for the buffer that is used
/// only by spv.Load and spv.SubgroupBlockReadINTEL.
/// The qualifier is added with OpMemberDecorate FuncParamAttr NoWrite.
/// This works only with spirv Capability "Kernel" (OpenCL backend).

class IntelGenOclSetAccessQualifiers
    : public IntelGenOclSetAccessQualifiersBase<
          IntelGenOclSetAccessQualifiers> {
public:
  /// Returns entry point function in module or nullptr if there is none.
  spirv::FuncOp getEntryPoint(spirv::ModuleOp module) {
    spirv::FuncOp func = nullptr;
    module.walk([&](spirv::FuncOp op) {
      if (!op->getAttr(spirv::getEntryPointABIAttrName()))
        return mlir::WalkResult::advance();
      func = op;
      return mlir::WalkResult::interrupt();
    });
    return func;
  }

  bool checkForReadOnly(mlir::Operation *op) {
    for (auto *user : op->getUsers()) {
      if (!(mlir::isa<spirv::LoadOp>(user) ||
            mlir::isa<spirv::SubgroupBlockReadINTELOp>(user)))
        return false;
    }
    return true;
  }

  void runOnOperation() {
    spirv::ModuleOp module = getOperation();
    spirv::FuncOp func = getEntryPoint(module);
    if (!func)
      return;

    for (mlir::BlockArgument arg : func.getArguments()) {
      auto readOnly = false;
      for (auto *op : arg.getUsers()) {
        // Case where argument is used directly by AccessChainOp
        if (mlir::isa<spirv::AccessChainOp>(op)) {
          readOnly = checkForReadOnly(op);
        } else if (mlir::isa<spirv::BitcastOp>(op)) {
          // Case where argument is casted before AccessChainOp
          for (auto *bitCastUser : op->getUsers()) {
            if (mlir::isa<spirv::AccessChainOp>(bitCastUser)) {
              readOnly = checkForReadOnly(bitCastUser);
            } else {
              readOnly = false;
              break;
            }
          }
        } else {
          // If something else then go to next argument immediately
          readOnly = false;
          break;
        }
      }
      // Add NoWrite decoration to the argument
      if (readOnly) {
        auto ptrType = arg.getType().dyn_cast<spirv::PointerType>();
        if (!ptrType)
          continue;

        auto structType =
            ptrType.getPointeeType().dyn_cast<spirv::StructType>();
        if (!ptrType)
          continue;

        mlir::SmallVector<mlir::Type, 4> memberTypes;
        mlir::SmallVector<spirv::StructType::OffsetInfo, 4> offsetInfo;
        mlir::SmallVector<spirv::StructType::MemberDecorationInfo, 4>
            memberDecorations;

        for (uint32_t i = 0; i < structType.getNumElements(); i++) {
          memberTypes.push_back(structType.getElementType(i));
          offsetInfo.push_back(structType.getMemberOffset(i));
        }
        structType.getMemberDecorations(memberDecorations);

        // TODO: create new enum in SPIRV dialect that would use actual enum and
        // not the magic number 6 (that is NoWrite)
        spirv::StructType::MemberDecorationInfo readOnlyDecor(
            0, 1, spirv::Decoration::FuncParamAttr, 6);
        memberDecorations.push_back(readOnlyDecor);

        auto newStructType =
            spirv::StructType::get(memberTypes, offsetInfo, memberDecorations);
        auto newPtrType =
            spirv::PointerType::get(newStructType, ptrType.getStorageClass());
        arg.setType(newPtrType);
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createSetAccessQualifiersPass() {
  return std::make_unique<IntelGenOclSetAccessQualifiers>();
}

} // namespace pmlc::target::intel_gen_ocl_spirv
