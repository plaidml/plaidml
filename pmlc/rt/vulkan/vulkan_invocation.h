// Vulkan invocation interface, originally from the LLVM project, and
// subsequently modified by Intel Corporation.
//
// Original copyright:
//
//===- VulkanRuntime.cpp - MLIR Vulkan runtime ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <memory>
#include <vector>

#include "pmlc/rt/vulkan/vulkan_device.h"

namespace pmlc::rt::vulkan {

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

// VulkanInvocation encapsulates a particular run of a network on a Vulkan
// device.  It's instantiated and managed from the JITted network code, using
// callbacks in wrappers.cc.
class VulkanInvocation {
public:
  VulkanInvocation();
  ~VulkanInvocation();

  void createLaunchKernelAction(uint8_t *shader, uint32_t size,
                                const char *entryPoint,
                                NumWorkGroups numWorkGroups);

  void setLaunchKernelAction();

  void addLaunchActionToSchedule();

  void createMemoryTransferAction(uint64_t src_index, uint64_t src_binding,
                                  uint64_t dst_index, uint64_t dst_binding);

  void createMemoryTransferAction(VkBuffer src, VkBuffer dst, size_t size);

  void submitCommandBuffers();

  /// Sets needed data for Vulkan device.
  void setResourceData(const ResourceData &resData);
  void setResourceData(const DescriptorSetIndex desIndex,
                       const BindingIndex bindIndex,
                       const VulkanHostMemoryBuffer &hostMemBuffer);

private:
  void setResourceStorageClassBindingMap(
      const ResourceStorageClassBindingMap &stClassData);

  void mapStorageClassToDescriptorType(mlir::spirv::StorageClass storageClass,
                                       VkDescriptorType &descriptorType);

  void mapStorageClassToBufferUsageFlag(mlir::spirv::StorageClass storageClass,
                                        VkBufferUsageFlagBits &bufferUsage);

  void checkResourceData();

  void createMemoryBuffers();
  void createShaderModule();
  void initDescriptorSetLayoutBindingMap();
  void createDescriptorSetLayout();
  void createPipelineLayout();
  void createComputePipeline();
  void createDescriptorPool();
  void allocateDescriptorSets();
  void setWriteDescriptors();
  void createSchedule();
  void submitCommandBuffersToQueue();
  void updateHostMemoryBuffers();

  std::vector<ActionPtr> schedule;
  std::shared_ptr<LaunchKernelAction> curr;
  std::shared_ptr<VulkanDevice> device;
  VkCommandPool commandPool;
  llvm::SmallVector<VkCommandBuffer, 1> commandBuffers;
  VkQueryPool timestampQueryPool;
  const uint32_t timestampQueryPoolSize{8192};
  uint32_t timestampQueryCount{2};
  uint32_t memoryTransferCount{0};
};

} // namespace pmlc::rt::vulkan
