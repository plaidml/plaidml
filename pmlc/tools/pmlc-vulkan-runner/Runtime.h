#ifndef PMLC_TOOLS_PMLC_VULKAN_RUNNER_RUNTIME_H_
#define PMLC_TOOLS_PMLC_VULKAN_RUNNER_RUNTIME_H_

#include <memory>
#include <random>
#include <string>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Dialect/SPIRV/SPIRVBinaryUtils.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"

namespace pmlc::vulkan {

using DescriptorSetIndex = uint32_t;
using BindingIndex = uint32_t;

// Struct containing information regarding to a host memory buffer.
struct VulkanHostMemoryBuffer {
  void* ptr{nullptr};
  uint64_t size{0};
};

// Struct containing the number of local workgroups to dispatch for each
// dimension.
struct NumWorkGroups {
  uint32_t x{1};
  uint32_t y{1};
  uint32_t z{1};
};

extern mlir::LogicalResult runOnVulkan(
    mlir::ModuleOp, llvm::DenseMap<DescriptorSetIndex, llvm::DenseMap<BindingIndex, VulkanHostMemoryBuffer>>&,
    const NumWorkGroups&);

class RuntimeTest {
 public:
  mlir::LogicalResult parseAndRunModule(llvm::StringRef sourceFile, NumWorkGroups numWorkGroups);

  std::unique_ptr<float[]> createResourceVarFloat(uint32_t descriptorSet, uint32_t binding, uint32_t elementCount);

  void destroyResourceVarFloat(VulkanHostMemoryBuffer& hostMemoryBuffer);

  VulkanHostMemoryBuffer FMul(VulkanHostMemoryBuffer& var1, VulkanHostMemoryBuffer& var2);

  VulkanHostMemoryBuffer FAdd(VulkanHostMemoryBuffer& var1, VulkanHostMemoryBuffer& var2);

  bool isEqualFloat(const VulkanHostMemoryBuffer& hostMemoryBuffer1, const VulkanHostMemoryBuffer& hostMemoryBuffer2);

  llvm::DenseMap<DescriptorSetIndex, llvm::DenseMap<BindingIndex, VulkanHostMemoryBuffer>> vars;
  std::random_device randomDevice;
};
}  // namespace pmlc::vulkan
#endif  // PMLC_TOOLS_PMLC_VULKAN_RUNNER_RUNTIME_H_
