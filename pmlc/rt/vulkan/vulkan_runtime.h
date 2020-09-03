//===- VulkanRuntime.cpp - MLIR Vulkan runtime ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares Vulkan runtime API.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>
#include <vector>

#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/IR/Module.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ToolOutputFile.h"

#include "volk.h" // NOLINT[build/include_subdir]

using DescriptorSetIndex = uint32_t;
using BindingIndex = uint32_t;

/// Struct containing information regarding to a device memory buffer.
struct VulkanDeviceMemoryBuffer {
  BindingIndex bindingIndex{0};
  VkDescriptorType descriptorType{VK_DESCRIPTOR_TYPE_MAX_ENUM};
  VkDescriptorBufferInfo bufferInfo{};
  VkBuffer buffer{VK_NULL_HANDLE};
  VkDeviceMemory deviceMemory{VK_NULL_HANDLE};
  size_t bufferSize{0};
};

/// Struct containing information regarding to a host memory buffer.
struct VulkanHostMemoryBuffer {
  /// Pointer to a host memory.
  void *ptr{nullptr};
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
using ResourceData =
    llvm::DenseMap<DescriptorSetIndex,
                   llvm::DenseMap<BindingIndex, VulkanHostMemoryBuffer>>;

/// StorageClass mapped into a descriptor set and a binding.
using ResourceStorageClassBindingMap =
    llvm::DenseMap<DescriptorSetIndex,
                   llvm::DenseMap<BindingIndex, mlir::spirv::StorageClass>>;

inline void emitVulkanError(const llvm::Twine &message, VkResult error) {
  llvm::errs()
      << message.concat(" failed with error code ").concat(llvm::Twine{error});
}

#define RETURN_ON_VULKAN_ERROR(result, msg)                                    \
  if ((result) != VK_SUCCESS) {                                                \
    emitVulkanError(msg, (result));                                            \
    return failure();                                                          \
  }

struct Action {
  virtual ~Action() {}
};

using ActionPtr = std::shared_ptr<Action>;

struct LaunchKernelAction : Action {
  /// Specifies VulkanDeviceMemoryBuffers divided into sets.
  llvm::DenseMap<DescriptorSetIndex,
                 llvm::SmallVector<VulkanDeviceMemoryBuffer, 1>>
      deviceMemoryBufferMap;

  /// Specifies shader module.
  VkShaderModule shaderModule;

  /// Specifies layout bindings.
  llvm::DenseMap<DescriptorSetIndex,
                 llvm::SmallVector<VkDescriptorSetLayoutBinding, 1>>
      descriptorSetLayoutBindingMap;

  /// Specifies layouts of descriptor sets.
  llvm::SmallVector<VkDescriptorSetLayout, 1> descriptorSetLayouts;
  VkPipelineLayout pipelineLayout;

  /// Specifies descriptor sets.
  llvm::SmallVector<VkDescriptorSet, 1> descriptorSets;

  /// Specifies a pool of descriptor set info, each descriptor set must have
  /// information such as type, index and amount of bindings.
  llvm::SmallVector<DescriptorSetInfo, 1> descriptorSetInfoPool;
  VkDescriptorPool descriptorPool;

  /// Computation pipeline.
  VkPipeline pipeline;

  //===--------------------------------------------------------------------===//
  // Vulkan execution context.
  //===--------------------------------------------------------------------===//

  NumWorkGroups workGroups;
  const char *entryPoint{nullptr};
  uint8_t *binary{nullptr};
  uint32_t binarySize{0};

  //===--------------------------------------------------------------------===//
  // Vulkan resource data and storage classes.
  //===--------------------------------------------------------------------===//

  ResourceData resourceData;
  ResourceStorageClassBindingMap resourceStorageClassData;

  llvm::SmallVector<VkBufferMemoryBarrier, 4> deps;
};

struct MemoryTransferAction : Action {
  VkBuffer src;
  VkBuffer dst;
  llvm::SmallVector<VkBufferCopy, 1> regions;
};

/// Vulkan runtime.
/// The purpose of this class is to run SPIR-V compute shader on Vulkan
/// device.
/// Before the run, user must provide and set resource data with descriptors,
/// SPIR-V shader, number of work groups and entry point. After the creation of
/// VulkanRuntime, special methods must be called in the following
/// sequence: initRuntime(), run(), updateHostMemoryBuffers(), destroy();
/// each method in the sequence returns succes or failure depends on the Vulkan
/// result code.
class VulkanRuntime {
public:
  VulkanRuntime() = default;
  VulkanRuntime(const VulkanRuntime &) = delete;
  VulkanRuntime &operator=(const VulkanRuntime &) = delete;

  mlir::LogicalResult init();
  mlir::LogicalResult destroy();
  mlir::LogicalResult createLaunchKernelAction(uint8_t *shader, uint32_t size,
                                               const char *entryPoint,
                                               NumWorkGroups numWorkGroups);
  mlir::LogicalResult setLaunchKernelAction();
  void addLaunchActionToSchedule();
  mlir::LogicalResult createMemoryTransferAction(uint64_t src_index,
                                                 uint64_t src_binding,
                                                 uint64_t dst_index,
                                                 uint64_t dst_binding);
  mlir::LogicalResult createMemoryTransferAction(VkBuffer src, VkBuffer dst,
                                                 size_t size);
  mlir::LogicalResult submitCommandBuffers();
  /// Sets needed data for Vulkan runtime.
  void setResourceData(const ResourceData &resData);
  void setResourceData(const DescriptorSetIndex desIndex,
                       const BindingIndex bindIndex,
                       const VulkanHostMemoryBuffer &hostMemBuffer);

private:
  //===--------------------------------------------------------------------===//
  // Pipeline creation methods.
  //===--------------------------------------------------------------------===//
  mlir::LogicalResult createInstance();
  mlir::LogicalResult createDevice();
  mlir::LogicalResult
  getBestComputeQueue(const VkPhysicalDevice &physicalDevice);
  mlir::LogicalResult createMemoryBuffers();
  mlir::LogicalResult createShaderModule();
  void initDescriptorSetLayoutBindingMap();
  mlir::LogicalResult createDescriptorSetLayout();
  mlir::LogicalResult createPipelineLayout();
  mlir::LogicalResult createComputePipeline();
  mlir::LogicalResult createDescriptorPool();
  mlir::LogicalResult allocateDescriptorSets();
  mlir::LogicalResult setWriteDescriptors();
  mlir::LogicalResult createCommandPool();
  mlir::LogicalResult checkResourceData();
  mlir::LogicalResult createSchedule();
  mlir::LogicalResult submitCommandBuffersToQueue();
  mlir::LogicalResult updateHostMemoryBuffers();
  void setResourceStorageClassBindingMap(
      const ResourceStorageClassBindingMap &stClassData);

  //===--------------------------------------------------------------------===//
  // Helper methods.
  //===--------------------------------------------------------------------===//

  /// Maps storage class to a descriptor type.
  mlir::LogicalResult
  mapStorageClassToDescriptorType(mlir::spirv::StorageClass storageClass,
                                  VkDescriptorType &descriptorType);

  /// Maps storage class to buffer usage flags.
  mlir::LogicalResult
  mapStorageClassToBufferUsageFlag(mlir::spirv::StorageClass storageClass,
                                   VkBufferUsageFlagBits &bufferUsage);

  mlir::LogicalResult countDeviceMemorySize();

  VkResult volkInitialized = VK_RESULT_MAX_ENUM;

  //===--------------------------------------------------------------------===//
  // Vulkan objects.
  //===--------------------------------------------------------------------===//
  VkInstance instance;
  VkDevice device;
  VkQueue queue;
  VkCommandPool commandPool;

  //===--------------------------------------------------------------------===//
  // Vulkan memory context.
  //===--------------------------------------------------------------------===//
  uint32_t queueFamilyIndex{0};
  uint32_t memoryTypeIndex{VK_MAX_MEMORY_TYPES};
  VkDeviceSize memorySize{0};

  std::vector<ActionPtr> schedule;
  std::shared_ptr<LaunchKernelAction> curr;
  llvm::SmallVector<VkCommandBuffer, 1> commandBuffers;
};
