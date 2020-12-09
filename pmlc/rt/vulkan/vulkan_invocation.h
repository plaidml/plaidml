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

class VulkanInvocation;

/// Struct containing information regarding to a device memory buffer.
struct VulkanDeviceMemoryBuffer {
  BindingIndex bindingIndex{0};
  VkDescriptorType descriptorType{VK_DESCRIPTOR_TYPE_MAX_ENUM};
  VkDescriptorBufferInfo bufferInfo{};
  VkBuffer buffer{VK_NULL_HANDLE};
  VkDeviceMemory deviceMemory{VK_NULL_HANDLE};
  size_t bufferSize{0};
};

/// A wrapper of VulkanDeviceMemoryBuffer.
struct vulkanBuffer {
  VulkanDeviceMemoryBuffer devBuffer;
  DescriptorSetIndex descriptorSet;
};

/// A wrapper of VkEvent.
struct vulkanEvent {
  VkEvent event{VK_NULL_HANDLE};
  VulkanInvocation *invocation = nullptr;
  bool bufferCopyEvent = false;
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
};

struct MemoryTransferAction : Action {
  VkBuffer src;
  VkBuffer dst;
  llvm::SmallVector<VkBufferCopy, 1> regions;
};

struct SetEventAction : Action {
  VkEvent event;
};

struct WaitEventsAction : Action {
  std::vector<VkEvent> events;
};

// VulkanInvocation encapsulates a particular run of a network on a Vulkan
// device.  It's instantiated and managed from the JITted network code, using
// callbacks in wrappers.cc.
class VulkanInvocation {
public:
  explicit VulkanInvocation(VulkanDevice *device);
  ~VulkanInvocation();

  vulkanBuffer *createMemoryBuffer(uint32_t bytes, void *hostPtr);
  vulkanEvent *scheduleLaunchKernelAction(uint32_t subgroupSize,
                                          uint8_t *shader, uint32_t size,
                                          const char *entryPoint,
                                          NumWorkGroups numWorkGroups,
                                          std::vector<void *> deviceBuffers);
  void run();
  VkEvent createSetEventAction();
  void createWaitEventsAction(std::vector<vulkanEvent *> &events);
  vulkanEvent *copyHostBufferToDevice(void *srcPtr, void *deviceBuffer);
  vulkanEvent *copyDeviceBufferToHost(void *hostPtr, void *deviceBuffer);
  void deallocDeviceBuffer(void *buffer);

private:
  void createQueryPool();
  void createSchedule();
  void freeScheduleResources();
  void freeCommandBuffers();
  void getQueryPoolResults();
  void submitCommandBuffersToQueue();
  void mapStorageClassToDescriptorType(mlir::spirv::StorageClass storageClass,
                                       VkDescriptorType &descriptorType);
  void mapStorageClassToBufferUsageFlag(mlir::spirv::StorageClass storageClass,
                                        VkBufferUsageFlagBits &bufferUsage);
  void allocateDescriptorSets(std::shared_ptr<LaunchKernelAction> &action);
  void createComputePipeline(std::shared_ptr<LaunchKernelAction> &action,
                             uint32_t subgroupSize);
  void createDescriptorPool(std::shared_ptr<LaunchKernelAction> &action);
  void createDescriptorSetLayout(std::shared_ptr<LaunchKernelAction> &action);
  void createPipelineLayout(std::shared_ptr<LaunchKernelAction> &action);
  void createShaderModule(std::shared_ptr<LaunchKernelAction> &action);
  void initDescriptorSetLayoutBindingMap(
      std::shared_ptr<LaunchKernelAction> &action);
  void setWriteDescriptors(std::shared_ptr<LaunchKernelAction> &action);

  std::vector<ActionPtr> schedule;
  std::vector<vulkanBuffer *> deviceBufferPool;
  std::shared_ptr<VulkanDevice> device;
  VkCommandPool commandPool;
  llvm::SmallVector<VkCommandBuffer, 1> commandBuffers;
  VkQueryPool timestampQueryPool;
  const uint32_t timestampQueryPoolSize{8192};
  uint32_t timestampQueryCount{0};
  double bufferTransferTime{0.0};
  uint32_t bufferTransferCount{0};
};

} // namespace pmlc::rt::vulkan
