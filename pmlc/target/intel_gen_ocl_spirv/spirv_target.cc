// Copyright 2020, Intel Corporation

#include "pmlc/target/intel_gen_ocl_spirv/pass_detail.h"
#include "pmlc/target/intel_gen_ocl_spirv/passes.h"

#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/Support/LLVM.h"

namespace pmlc::target::intel_gen_ocl_spirv {
namespace spirv = mlir::spirv;

namespace {

class IntelGenOclAddSpirvTarget
    : public IntelGenOclAddSpirvTargetBase<IntelGenOclAddSpirvTarget> {
public:
  void runOnOperation() {
    auto target_env = getOperation().getAttrOfType<spirv::TargetEnvAttr>(
        spirv::getTargetEnvAttrName());
    if (!target_env) {
      auto triple = spirv::VerCapExtAttr::get(
          spirv::Version::V_1_5,
          {spirv::Capability::Kernel, spirv::Capability::Addresses,
           spirv::Capability::Groups, spirv::Capability::SubgroupDispatch,
           spirv::Capability::Int64, spirv::Capability::Int16,
           spirv::Capability::Int8, spirv::Capability::Float64,
           spirv::Capability::Float16, spirv::Capability::GroupNonUniformBallot,
           spirv::Capability::SubgroupBufferBlockIOINTEL},
          mlir::ArrayRef<spirv::Extension>(
              spirv::Extension::SPV_INTEL_subgroups),
          &getContext());
      getOperation().setAttr(
          spirv::getTargetEnvAttrName(),
          spirv::TargetEnvAttr::get(
              triple, spirv::Vendor::Unknown, spirv::DeviceType::Unknown,
              spirv::TargetEnvAttr::kUnknownDeviceID,
              spirv::getDefaultResourceLimits(&getContext())));
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createAddSpirvTargetPass() {
  return std::make_unique<IntelGenOclAddSpirvTarget>();
}

} // namespace pmlc::target::intel_gen_ocl_spirv
