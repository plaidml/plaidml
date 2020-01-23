#pragma once

#include "pmlc/tools/pmlc-vulkan-runner/vulkan_pfn.h"

#include "mlir/Analysis/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/IR/Module.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/StringExtras.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;  // NOLINT[build/namespaces]
using DescriptorSetIndex = uint32_t;
using BindingIndex = uint32_t;

namespace pmlc::vulkan {
/// Struct containing information regarding to a device memory buffer.
struct VulkanDeviceMemoryBuffer {
  BindingIndex bindingIndex{0};
  VkDescriptorType descriptorType{VK_DESCRIPTOR_TYPE_MAX_ENUM};
  VkDescriptorBufferInfo bufferInfo{VK_NULL_HANDLE};
  VkBuffer buffer{VK_NULL_HANDLE};
  VkDeviceMemory deviceMemory{VK_NULL_HANDLE};
};

/// Struct containing information regarding to a host memory buffer.
struct VulkanHostMemoryBuffer {
  /// Pointer to a host memory.
  void* ptr{nullptr};
  /// Size of a host memory in bytes.
  uint32_t size{0};
};

/// Struct containing the number of local workgroups to dispatch for each
/// dimension.
struct NumWorkGroups {
  uint32_t x{1};
  uint32_t y{1};
  uint32_t z{1};
};

/// Struct containing information regarding a descriptor set.
struct DescriptorSetInfo {
  /// Index of a descriptor set in descriptor sets.
  DescriptorSetIndex descriptorSet{0};
  /// Number of desriptors in a set.
  uint32_t descriptorSize{0};
  /// Type of a descriptor set.
  VkDescriptorType descriptorType{VK_DESCRIPTOR_TYPE_MAX_ENUM};
};

/// VulkanHostMemoryBuffer mapped into a descriptor set and a binding.
using ResourceData = llvm::DenseMap<DescriptorSetIndex, llvm::DenseMap<BindingIndex, VulkanHostMemoryBuffer>>;

/// StorageClass mapped into a descriptor set and a binding.
using ResourceStorageClassData =
    llvm::DenseMap<DescriptorSetIndex, llvm::DenseMap<BindingIndex, mlir::spirv::StorageClass>>;

LogicalResult runOnVulkan(mlir::ModuleOp module, ResourceData& resourceData, const NumWorkGroups& numWorkGroups);

}  // namespace pmlc::vulkan
