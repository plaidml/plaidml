// Vulkan invocation implementation, originally from the LLVM project, and
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

#include "pmlc/rt/vulkan/vulkan_invocation.h"

#include <chrono>

#include "llvm/Support/FormatVariadic.h"

#include "pmlc/rt/vulkan/vulkan_error.h"
#include "pmlc/util/logging.h"

namespace pmlc::rt::vulkan {

typedef std::chrono::high_resolution_clock Clock;

VulkanInvocation::VulkanInvocation(VulkanDevice *device)
    : device{device->shared_from_this()} {
  createQueryPool();
}

VulkanInvocation::~VulkanInvocation() {
  if (device->getTimestampValidBits()) {
    getQueryPoolResults();
  }
  vkDestroyCommandPool(device->getDevice(), commandPool, nullptr);
  vkDestroyQueryPool(device->getDevice(), timestampQueryPool,
                     /*allocator=*/nullptr);
}

void VulkanInvocation::createQueryPool() {
  VkCommandPoolCreateInfo commandPoolCreateInfo = {};
  commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  commandPoolCreateInfo.pNext = nullptr;
  commandPoolCreateInfo.flags = 0;
  commandPoolCreateInfo.queueFamilyIndex = device->getQueueFamilyIndex();
  throwOnVulkanError(vkCreateCommandPool(device->getDevice(),
                                         &commandPoolCreateInfo, 0,
                                         &commandPool),
                     "vkCreateCommandPool");
  VkQueryPoolCreateInfo queryPoolCreateInfo = {};
  queryPoolCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  queryPoolCreateInfo.pNext = nullptr;
  queryPoolCreateInfo.flags = 0;
  queryPoolCreateInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
  queryPoolCreateInfo.queryCount = timestampQueryPoolSize;
  queryPoolCreateInfo.pipelineStatistics = 0;
  throwOnVulkanError(vkCreateQueryPool(device->getDevice(),
                                       &queryPoolCreateInfo, /*allocator=*/
                                       nullptr, &timestampQueryPool),
                     "vkCreateQueryPool");
}

vulkanEvent *VulkanInvocation::scheduleLaunchKernelAction(
    uint32_t subgroupSize, uint8_t *shader, uint32_t size,
    const char *entryPoint, NumWorkGroups numWorkGroups,
    std::vector<void *> deviceBuffers) {

  auto action = std::make_shared<LaunchKernelAction>();
  action->binary = shader;
  action->binarySize = size;
  action->entryPoint = entryPoint;
  action->workGroups = numWorkGroups;

  for (auto bufferPtr : deviceBuffers) {
    auto buffer = static_cast<vulkanBuffer *>(bufferPtr);
    action->deviceMemoryBufferMap[buffer->descriptorSet].push_back(
        buffer->devBuffer);
  }

  // Create shader module
  createShaderModule(action);

  // Descriptor bindings divided into sets. Each descriptor binding
  // must have a layout binding attached into a descriptor set layout.
  // Each layout set must be binded into a pipeline layout.
  initDescriptorSetLayoutBindingMap(action);
  createDescriptorSetLayout(action);
  createPipelineLayout(action);
  createComputePipeline(action, subgroupSize);

  // Each descriptor set must be allocated from a descriptor pool.
  createDescriptorPool(action);
  allocateDescriptorSets(action);
  setWriteDescriptors(action);
  schedule.push_back(action);

  return new vulkanEvent{createSetEventAction(), this, false};
}

VkEvent VulkanInvocation::createSetEventAction() {
  // Create SetEventAction and push to schedule
  VkEvent Event;
  VkEventCreateInfo eventCreateInfo = {};
  eventCreateInfo.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO;
  eventCreateInfo.pNext = nullptr;
  eventCreateInfo.flags = 0;
  vkCreateEvent(device->getDevice(), &eventCreateInfo, nullptr, &Event);

  auto setEventAction = std::make_shared<SetEventAction>();
  setEventAction->event = Event;
  schedule.push_back(setEventAction);
  return Event;
}

void VulkanInvocation::createWaitEventsAction(
    std::vector<vulkanEvent *> &vulkanEvents) {
  auto waitEvents = std::make_shared<WaitEventsAction>();
  for (auto valkanEvent : vulkanEvents) {
    if (valkanEvent->bufferCopyEvent == false) {
      waitEvents->events.push_back(valkanEvent->event);
    }
  }
  schedule.push_back(waitEvents);
}

void VulkanInvocation::getQueryPoolResults() {
  using fp_milliseconds =
      std::chrono::duration<double, std::chrono::milliseconds::period>;
  using fp_nanoseconds =
      std::chrono::duration<double, std::chrono::nanoseconds::period>;

  uint64_t *results = reinterpret_cast<uint64_t *>(
      calloc(timestampQueryCount, sizeof(uint64_t)));
  vkGetQueryPoolResults(device->getDevice(), timestampQueryPool,
                        /*firstQuery=*/0,
                        /*queryCount=*/timestampQueryCount,
                        /*dataSize=*/timestampQueryCount * sizeof(uint64_t),
                        results,
                        /*stride=*/sizeof(uint64_t),
                        (VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));

  fp_nanoseconds total_kernel_ns{0};
  for (uint32_t i = 0; i < timestampQueryCount; i += 2) {
    fp_nanoseconds kernel_ns{(results[i + 1] - results[i]) *
                             device->getTimestampPeriod()};

    IVLOG(2, "  Kernel execute time: " << fp_milliseconds(kernel_ns).count()
                                       << "ms");
    total_kernel_ns += kernel_ns;
  }

  if (timestampQueryCount == timestampQueryPoolSize) {
    IVLOG(1,
          "WARNING: Ran out of space in the timestamp query pool which has "
          "size = "
              << timestampQueryPoolSize
              << "; consider increasing the size of the timestamp query pool.");
  }

  IVLOG(1, "Total Vulkan kernels: " << timestampQueryCount / 2);
  IVLOG(1, "Total Vulkan kernel execute time: "
               << fp_milliseconds(total_kernel_ns).count() << "ms");

  IVLOG(1, "Total Vulkan memory transfers: " << bufferTransferCount);
  IVLOG(1, "Total Vulkan memory transfer time: " << bufferTransferTime << "ms");

  device->execTimeInMS =
      fp_milliseconds(total_kernel_ns).count() + bufferTransferTime;
}

void VulkanInvocation::freeScheduleResources() {
  for (const auto &action : schedule) {
    if (auto kernel = std::dynamic_pointer_cast<LaunchKernelAction>(action)) {
      vkFreeDescriptorSets(device->getDevice(), kernel->descriptorPool,
                           kernel->descriptorSets.size(),
                           kernel->descriptorSets.data());
      vkDestroyDescriptorPool(device->getDevice(), kernel->descriptorPool,
                              nullptr);
      vkDestroyPipeline(device->getDevice(), kernel->pipeline, nullptr);
      vkDestroyPipelineLayout(device->getDevice(), kernel->pipelineLayout,
                              nullptr);
      for (auto &descriptorSetLayout : kernel->descriptorSetLayouts) {
        vkDestroyDescriptorSetLayout(device->getDevice(), descriptorSetLayout,
                                     nullptr);
      }
      vkDestroyShaderModule(device->getDevice(), kernel->shaderModule, nullptr);
    }
  }
  schedule.clear();
}

void VulkanInvocation::freeCommandBuffers() {
  vkFreeCommandBuffers(device->getDevice(), commandPool, commandBuffers.size(),
                       commandBuffers.data());
  commandBuffers.clear();
}

void VulkanInvocation::run() {
  createSchedule();
  submitCommandBuffersToQueue();
  throwOnVulkanError(vkQueueWaitIdle(device->getQueue()), "vkQueueWaitIdle");
  freeCommandBuffers();
  freeScheduleResources();
}

void VulkanInvocation::mapStorageClassToDescriptorType(
    mlir::spirv::StorageClass storageClass, VkDescriptorType &descriptorType) {
  switch (storageClass) {
  case mlir::spirv::StorageClass::StorageBuffer:
    descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    break;
  case mlir::spirv::StorageClass::Uniform:
    descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    break;
  default:
    throw std::runtime_error{"unsupported storage class"};
  }
}

void VulkanInvocation::mapStorageClassToBufferUsageFlag(
    mlir::spirv::StorageClass storageClass,
    VkBufferUsageFlagBits &bufferUsage) {
  switch (storageClass) {
  case mlir::spirv::StorageClass::StorageBuffer:
    bufferUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    break;
  case mlir::spirv::StorageClass::Uniform:
    bufferUsage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    break;
  default:
    throw std::runtime_error{"unsupported storage class"};
  }
}

vulkanEvent *VulkanInvocation::copyDeviceBufferToHost(void *hostPtr,
                                                      void *deviceBuffer) {
  auto start = Clock::now();
  void *payload;
  auto vulkanDeviceMemoryBuffer =
      static_cast<vulkanBuffer *>(deviceBuffer)->devBuffer;
  auto deviceMemoryBuffer = vulkanDeviceMemoryBuffer.deviceMemory;
  auto bufferSize = vulkanDeviceMemoryBuffer.bufferSize;
  throwOnVulkanError(vkMapMemory(device->getDevice(), deviceMemoryBuffer, 0,
                                 bufferSize, 0,
                                 reinterpret_cast<void **>(&payload)),
                     "vkMapMemory");
  std::memcpy(hostPtr, payload, bufferSize);
  vkUnmapMemory(device->getDevice(), deviceMemoryBuffer);
  auto end = Clock::now();
  bufferTransferTime +=
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  bufferTransferCount++;
  return new vulkanEvent{VK_NULL_HANDLE, this, true};
}

vulkanEvent *VulkanInvocation::copyHostBufferToDevice(void *srcPtr,
                                                      void *deviceBuffer) {
  auto start = Clock::now();
  void *payload;
  auto vulkanDeviceMemoryBuffer =
      static_cast<vulkanBuffer *>(deviceBuffer)->devBuffer;
  auto deviceMemoryBuffer = vulkanDeviceMemoryBuffer.deviceMemory;
  auto bufferSize = vulkanDeviceMemoryBuffer.bufferSize;
  throwOnVulkanError(vkMapMemory(device->getDevice(), deviceMemoryBuffer, 0,
                                 bufferSize, 0, &payload),
                     "vkMapMemory");
  std::memcpy(payload, srcPtr, bufferSize);
  vkUnmapMemory(device->getDevice(), deviceMemoryBuffer);
  auto end = Clock::now();
  bufferTransferTime +=
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  bufferTransferCount++;
  return new vulkanEvent{VK_NULL_HANDLE, this, true};
}

void VulkanInvocation::deallocDeviceBuffer(void *buffer) {
  auto vulkanDeviceMemoryBuffer =
      static_cast<vulkanBuffer *>(buffer)->devBuffer;
  vkFreeMemory(device->getDevice(), vulkanDeviceMemoryBuffer.deviceMemory,
               nullptr);
  vkDestroyBuffer(device->getDevice(), vulkanDeviceMemoryBuffer.buffer,
                  nullptr);
}

vulkanBuffer *VulkanInvocation::createMemoryBuffer(uint32_t bytes,
                                                   void *hostPtr) {
  vulkanBuffer *vulkanbuffer = new vulkanBuffer();
  VulkanDeviceMemoryBuffer &memoryBuffer = vulkanbuffer->devBuffer;
  VkDescriptorType descriptorType = {};
  VkBufferUsageFlagBits bufferUsageSrc = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  VkBufferUsageFlagBits bufferUsageDst = VK_BUFFER_USAGE_TRANSFER_DST_BIT;

  const auto resourceStorageClassBinding =
      mlir::spirv::StorageClass::StorageBuffer;

  mapStorageClassToDescriptorType(resourceStorageClassBinding, descriptorType);
  mapStorageClassToBufferUsageFlag(resourceStorageClassBinding, bufferUsageSrc);
  mapStorageClassToBufferUsageFlag(resourceStorageClassBinding, bufferUsageDst);
  // Set descriptor type for the specific device memory buffer.
  memoryBuffer.descriptorType = descriptorType;
  const auto bufferSize = bytes;

  // Specify memory allocation info.
  VkMemoryAllocateInfo memoryAllocateInfo = {};
  memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  memoryAllocateInfo.pNext = nullptr;
  memoryAllocateInfo.allocationSize = bufferSize;
  memoryAllocateInfo.memoryTypeIndex = device->getMemoryTypeIndex();

  // Allocate device memory.
  throwOnVulkanError(vkAllocateMemory(device->getDevice(), &memoryAllocateInfo,
                                      0, &memoryBuffer.deviceMemory),
                     "vkAllocateMemory");

  VkBufferCreateInfo bufferCreateInfo = {};
  bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferCreateInfo.pNext = nullptr;
  bufferCreateInfo.flags = 0;
  bufferCreateInfo.size = bufferSize;
  bufferCreateInfo.usage = bufferUsageSrc | bufferUsageDst;
  bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  bufferCreateInfo.queueFamilyIndexCount = 1;
  auto queueFamilyIndex = device->getQueueFamilyIndex();
  bufferCreateInfo.pQueueFamilyIndices = &queueFamilyIndex;
  throwOnVulkanError(vkCreateBuffer(device->getDevice(), &bufferCreateInfo, 0,
                                    &memoryBuffer.buffer),
                     "vkCreateBuffer");
  // Bind buffer and device memory.
  throwOnVulkanError(vkBindBufferMemory(device->getDevice(),
                                        memoryBuffer.buffer,
                                        memoryBuffer.deviceMemory, 0),
                     "vkBindBufferMemory");

  memoryBuffer.bufferSize = bufferSize;

  // Update buffer info.
  memoryBuffer.bufferInfo.buffer = memoryBuffer.buffer;
  memoryBuffer.bufferInfo.offset = 0;
  memoryBuffer.bufferInfo.range = VK_WHOLE_SIZE;

  DescriptorSetIndex index = 0;
  vulkanbuffer->descriptorSet = index;
  deviceBufferPool.push_back(vulkanbuffer);
  return vulkanbuffer;
}

void VulkanInvocation::createShaderModule(
    std::shared_ptr<LaunchKernelAction> &action) {
  VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
  shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shaderModuleCreateInfo.pNext = nullptr;
  shaderModuleCreateInfo.flags = 0;
  // Set size in bytes.
  shaderModuleCreateInfo.codeSize = action->binarySize;
  // Set pointer to the binary shader.
  shaderModuleCreateInfo.pCode = reinterpret_cast<uint32_t *>(action->binary);
  throwOnVulkanError(vkCreateShaderModule(device->getDevice(),
                                          &shaderModuleCreateInfo, 0,
                                          &action->shaderModule),
                     "vkCreateShaderModule");
}

void VulkanInvocation::initDescriptorSetLayoutBindingMap(
    std::shared_ptr<LaunchKernelAction> &action) {
  for (const auto &deviceMemoryBufferMapPair : action->deviceMemoryBufferMap) {
    llvm::SmallVector<VkDescriptorSetLayoutBinding, 1>
        descriptorSetLayoutBindings;
    const auto &deviceMemoryBuffers = deviceMemoryBufferMapPair.second;
    const auto descriptorSetIndex = deviceMemoryBufferMapPair.first;

    // Create a layout binding for each descriptor.
    for (size_t i = 0; i < deviceMemoryBuffers.size(); i++) {
      VkDescriptorSetLayoutBinding descriptorSetLayoutBinding = {};
      descriptorSetLayoutBinding.binding = i;
      descriptorSetLayoutBinding.descriptorType =
          deviceMemoryBuffers[i].descriptorType;
      descriptorSetLayoutBinding.descriptorCount = 1;
      descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      descriptorSetLayoutBinding.pImmutableSamplers = 0;
      descriptorSetLayoutBindings.push_back(descriptorSetLayoutBinding);
    }
    action->descriptorSetLayoutBindingMap[descriptorSetIndex] =
        descriptorSetLayoutBindings;
  }
}

void VulkanInvocation::createDescriptorSetLayout(
    std::shared_ptr<LaunchKernelAction> &action) {
  for (const auto &deviceMemoryBufferMapPair : action->deviceMemoryBufferMap) {
    const auto descriptorSetIndex = deviceMemoryBufferMapPair.first;
    const auto &deviceMemoryBuffers = deviceMemoryBufferMapPair.second;
    // Each descriptor in a descriptor set must be the same type.
    VkDescriptorType descriptorType =
        deviceMemoryBuffers.front().descriptorType;
    const uint32_t descriptorSize = deviceMemoryBuffers.size();
    const auto descriptorSetLayoutBindingIt =
        action->descriptorSetLayoutBindingMap.find(descriptorSetIndex);

    if (descriptorSetLayoutBindingIt ==
        action->descriptorSetLayoutBindingMap.end()) {
      throw std::runtime_error{llvm::formatv(
          "cannot find layout bindings for the set with number: {0}",
          descriptorSetIndex)};
    }

    const auto &descriptorSetLayoutBindings =
        descriptorSetLayoutBindingIt->second;
    // Create descriptor set layout.
    VkDescriptorSetLayout descriptorSetLayout = {};
    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};

    descriptorSetLayoutCreateInfo.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.pNext = nullptr;
    descriptorSetLayoutCreateInfo.flags = 0;
    // Amount of descriptor bindings in a layout set.
    descriptorSetLayoutCreateInfo.bindingCount =
        descriptorSetLayoutBindings.size();
    descriptorSetLayoutCreateInfo.pBindings =
        descriptorSetLayoutBindings.data();
    throwOnVulkanError(vkCreateDescriptorSetLayout(
                           device->getDevice(), &descriptorSetLayoutCreateInfo,
                           0, &descriptorSetLayout),
                       "vkCreateDescriptorSetLayout");

    action->descriptorSetLayouts.push_back(descriptorSetLayout);
    action->descriptorSetInfoPool.push_back(
        {descriptorSetIndex, descriptorSize, descriptorType});
  }
}

void VulkanInvocation::createPipelineLayout(
    std::shared_ptr<LaunchKernelAction> &action) {
  // Associate descriptor sets with a pipeline layout.
  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
  pipelineLayoutCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutCreateInfo.pNext = nullptr;
  pipelineLayoutCreateInfo.flags = 0;
  pipelineLayoutCreateInfo.setLayoutCount = action->descriptorSetLayouts.size();
  pipelineLayoutCreateInfo.pSetLayouts = action->descriptorSetLayouts.data();
  pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
  pipelineLayoutCreateInfo.pPushConstantRanges = 0;
  throwOnVulkanError(vkCreatePipelineLayout(device->getDevice(),
                                            &pipelineLayoutCreateInfo, 0,
                                            &action->pipelineLayout),
                     "vkCreatePipelineLayout");
}

void VulkanInvocation::createComputePipeline(
    std::shared_ptr<LaunchKernelAction> &action, uint32_t subgroupSize) {
  VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT subgroupSizeInfo;
  subgroupSizeInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  subgroupSizeInfo.requiredSubgroupSize = subgroupSize;
  subgroupSizeInfo.pNext = NULL;

  VkPipelineShaderStageCreateInfo stageInfo = {};
  stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  if (device->isExtensionSupported("VK_EXT_subgroup_size_control"))
    stageInfo.pNext = &subgroupSizeInfo;
  else
    stageInfo.pNext = NULL;
  stageInfo.pNext = nullptr;
  stageInfo.flags = 0;
  stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stageInfo.module = action->shaderModule;
  // Set entry point.
  stageInfo.pName = action->entryPoint;
  stageInfo.pSpecializationInfo = 0;

  VkComputePipelineCreateInfo computePipelineCreateInfo = {};
  computePipelineCreateInfo.sType =
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  computePipelineCreateInfo.pNext = nullptr;
  computePipelineCreateInfo.flags = 0;
  computePipelineCreateInfo.stage = stageInfo;
  computePipelineCreateInfo.layout = action->pipelineLayout;
  computePipelineCreateInfo.basePipelineHandle = 0;
  computePipelineCreateInfo.basePipelineIndex = 0;
  throwOnVulkanError(vkCreateComputePipelines(device->getDevice(), 0, 1,
                                              &computePipelineCreateInfo, 0,
                                              &action->pipeline),
                     "vkCreateComputePipelines");
}

void VulkanInvocation::createDescriptorPool(
    std::shared_ptr<LaunchKernelAction> &action) {
  llvm::SmallVector<VkDescriptorPoolSize, 1> descriptorPoolSizes;
  for (const auto &descriptorSetInfo : action->descriptorSetInfoPool) {
    // For each descriptor set populate descriptor pool size.
    VkDescriptorPoolSize descriptorPoolSize = {};
    descriptorPoolSize.type = descriptorSetInfo.descriptorType;
    descriptorPoolSize.descriptorCount = descriptorSetInfo.descriptorSize;
    descriptorPoolSizes.push_back(descriptorPoolSize);
  }

  VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
  descriptorPoolCreateInfo.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  descriptorPoolCreateInfo.pNext = nullptr;
  descriptorPoolCreateInfo.flags = 0;
  descriptorPoolCreateInfo.maxSets = descriptorPoolSizes.size();
  descriptorPoolCreateInfo.poolSizeCount = descriptorPoolSizes.size();
  descriptorPoolCreateInfo.pPoolSizes = descriptorPoolSizes.data();
  throwOnVulkanError(vkCreateDescriptorPool(device->getDevice(),
                                            &descriptorPoolCreateInfo, 0,
                                            &action->descriptorPool),
                     "vkCreateDescriptorPool");
}

void VulkanInvocation::allocateDescriptorSets(
    std::shared_ptr<LaunchKernelAction> &action) {
  VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
  // Size of descriptor sets and descriptor layout sets is the same.
  action->descriptorSets.resize(action->descriptorSetLayouts.size());
  descriptorSetAllocateInfo.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  descriptorSetAllocateInfo.pNext = nullptr;
  descriptorSetAllocateInfo.descriptorPool = action->descriptorPool;
  descriptorSetAllocateInfo.descriptorSetCount =
      action->descriptorSetLayouts.size();
  descriptorSetAllocateInfo.pSetLayouts = action->descriptorSetLayouts.data();
  throwOnVulkanError(vkAllocateDescriptorSets(device->getDevice(),
                                              &descriptorSetAllocateInfo,
                                              action->descriptorSets.data()),
                     "vkAllocateDescriptorSets");
}

void VulkanInvocation::setWriteDescriptors(
    std::shared_ptr<LaunchKernelAction> &action) {
  if (action->descriptorSets.size() != action->descriptorSetInfoPool.size()) {
    throw std::runtime_error{
        "Each descriptor set must have descriptor set information"};
  }
  // For each descriptor set.
  auto descriptorSetIt = action->descriptorSets.begin();
  // Each descriptor set is associated with descriptor set info.
  for (const auto &descriptorSetInfo : action->descriptorSetInfoPool) {
    // For each device memory buffer in the descriptor set.
    const auto &deviceMemoryBuffers =
        action->deviceMemoryBufferMap[descriptorSetInfo.descriptorSet];
    unsigned bindindex = 0;
    for (const auto &memoryBuffer : deviceMemoryBuffers) {
      // Structure describing descriptor sets to write to.
      VkWriteDescriptorSet wSet = {};
      wSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      wSet.pNext = nullptr;
      // Descriptor set.
      wSet.dstSet = *descriptorSetIt;
      wSet.dstBinding = bindindex++;
      wSet.dstArrayElement = 0;
      wSet.descriptorCount = 1;
      wSet.descriptorType = memoryBuffer.descriptorType;
      wSet.pImageInfo = nullptr;
      wSet.pBufferInfo = &memoryBuffer.bufferInfo;
      wSet.pTexelBufferView = nullptr;
      vkUpdateDescriptorSets(device->getDevice(), 1, &wSet, 0, nullptr);
    }
    // Increment descriptor set iterator.
    ++descriptorSetIt;
  }
}

void VulkanInvocation::createSchedule() {
  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.pNext = nullptr;
  allocInfo.commandPool = commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer;
  throwOnVulkanError(
      vkAllocateCommandBuffers(device->getDevice(), &allocInfo, &commandBuffer),
      "vkAllocateCommandBuffers");

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.pNext = nullptr;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  beginInfo.pInheritanceInfo = nullptr;

  throwOnVulkanError(vkBeginCommandBuffer(commandBuffer, &beginInfo),
                     "vkBeginCommandBuffer");

  for (const auto &action : schedule) {
    if (auto kernel = std::dynamic_pointer_cast<LaunchKernelAction>(action)) {
      if (device->getTimestampValidBits() &&
          timestampQueryCount < timestampQueryPoolSize) {
        vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            timestampQueryPool, timestampQueryCount++);
      }
      vkCmdBindPipeline(commandBuffer,
                        /*pipelineBindPoint=*/VK_PIPELINE_BIND_POINT_COMPUTE,
                        kernel->pipeline);
      vkCmdBindDescriptorSets(
          commandBuffer,
          /*pipelineBindPoint=*/VK_PIPELINE_BIND_POINT_COMPUTE,
          /*layout=*/kernel->pipelineLayout,
          /*firstSet=*/0,
          /*descriptorSetCount=*/kernel->descriptorSets.size(),
          /*pDescriptorSets=*/kernel->descriptorSets.data(),
          /*dynamicOffsetCount=*/0,
          /*pDynamicOffsets=*/0);
      vkCmdDispatch(commandBuffer,
                    /*groupCountX=*/kernel->workGroups.x,
                    /*groupCountY=*/kernel->workGroups.y,
                    /*groupCountZ=*/kernel->workGroups.z);
      if (device->getTimestampValidBits() &&
          timestampQueryCount < timestampQueryPoolSize) {
        vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            timestampQueryPool, timestampQueryCount++);
      }
    }
    if (auto xfer = std::dynamic_pointer_cast<MemoryTransferAction>(action)) {
      vkCmdCopyBuffer(commandBuffer,
                      /*srcBuffer=*/xfer->src,
                      /*dstBuffer=*/xfer->dst,
                      /*regionCount=*/xfer->regions.size(),
                      /*pRegions=*/xfer->regions.data());
    }
    if (auto set = std::dynamic_pointer_cast<SetEventAction>(action)) {
      vkCmdSetEvent(commandBuffer, set->event,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    }
    if (auto wait = std::dynamic_pointer_cast<WaitEventsAction>(action)) {
      vkCmdWaitEvents(commandBuffer, wait->events.size(), wait->events.data(),
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                      /*memoryBarrierCount=*/0,
                      /*pMemoryBarriers=*/nullptr,
                      /*bufferMemoryBarrierCount=*/0,
                      /*pBufferMemoryBarriers=*/nullptr,
                      /*imageMemoryBarrierCount=*/0,
                      /*pImageMemoryBarriers=*/nullptr);
    }
  }
  throwOnVulkanError(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");
  commandBuffers.push_back(commandBuffer);
}

void VulkanInvocation::submitCommandBuffersToQueue() {
  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.pNext = nullptr;
  submitInfo.waitSemaphoreCount = 0;
  submitInfo.pWaitSemaphores = 0;
  submitInfo.pWaitDstStageMask = 0;
  submitInfo.commandBufferCount = commandBuffers.size();
  submitInfo.pCommandBuffers = commandBuffers.data();
  submitInfo.signalSemaphoreCount = 0;
  submitInfo.pSignalSemaphores = nullptr;
  throwOnVulkanError(vkQueueSubmit(device->getQueue(), 1, &submitInfo, 0),
                     "vkQueueSubmit");
}

} // namespace pmlc::rt::vulkan
