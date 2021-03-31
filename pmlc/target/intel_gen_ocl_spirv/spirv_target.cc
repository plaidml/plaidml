// Copyright 2020, Intel Corporation

#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Support/LLVM.h"

#include "pmlc/target/intel_gen_ocl_spirv/pass_detail.h"
#include "pmlc/target/intel_gen_ocl_spirv/passes.h"
#include "pmlc/util/logging.h"

namespace pmlc::target::intel_gen_ocl_spirv {

using namespace mlir; // NOLINT

namespace {

class IntelGenOclAddSpirvTarget
    : public IntelGenOclAddSpirvTargetBase<IntelGenOclAddSpirvTarget> {
public:
  IntelGenOclAddSpirvTarget() = default;
  explicit IntelGenOclAddSpirvTarget(unsigned spirvVersion) {
    this->spirvVersion = spirvVersion;
  }
  void runOnOperation() {
    auto target_env = getOperation()->getAttrOfType<spirv::TargetEnvAttr>(
        spirv::getTargetEnvAttrName());
    if (!target_env) {
      IVLOG(3, "SPIR-V Version = " << spirvVersion);
      auto version = spirv::Version::V_1_5;
      if (spirvVersion == 120) {
        version = spirv::Version::V_1_2;
      }
      auto triple = spirv::VerCapExtAttr::get(
          version,
          {spirv::Capability::Kernel, spirv::Capability::Addresses,
           spirv::Capability::Groups, spirv::Capability::SubgroupDispatch,
           spirv::Capability::Int64, spirv::Capability::Int16,
           spirv::Capability::Int8, spirv::Capability::Float64,
           spirv::Capability::Float16, spirv::Capability::Vector16,
           spirv::Capability::GroupNonUniformBallot,
           spirv::Capability::SubgroupBufferBlockIOINTEL},
          mlir::ArrayRef<spirv::Extension>(
              spirv::Extension::SPV_INTEL_subgroups),
          &getContext());
      getOperation()->setAttr(
          spirv::getTargetEnvAttrName(),
          spirv::TargetEnvAttr::get(
              triple, spirv::Vendor::Unknown, spirv::DeviceType::Unknown,
              spirv::TargetEnvAttr::kUnknownDeviceID,
              spirv::getDefaultResourceLimits(&getContext())));
    }
  }
};

} // namespace

std::unique_ptr<Pass> createAddSpirvTargetPass() {
  return std::make_unique<IntelGenOclAddSpirvTarget>();
}

std::unique_ptr<Pass> createAddSpirvTargetPass(unsigned spirvVersion) {
  return std::make_unique<IntelGenOclAddSpirvTarget>(spirvVersion);
}

} // namespace pmlc::target::intel_gen_ocl_spirv
