//===- VulkanRuntime.cpp - MLIR Vulkan runtime ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a library for running a module on a Vulkan device.
// Implements a Vulkan runtime.
//
//===----------------------------------------------------------------------===//

#include "pmlc/rt/vulkan/vulkan_runtime.h"

#include <memory>
#include <vector>

#include <iostream>

using namespace mlir; // NOLINT[build/namespaces]

void VulkanRuntime::setNumWorkGroups(const NumWorkGroups &numberWorkGroups) {
  auto kernel = std::dynamic_pointer_cast<LaunchKernelAction>(schedule.back());
  kernel->workGroups = numberWorkGroups;
}

void VulkanRuntime::setResourceStorageClassBindingMap(
    const ResourceStorageClassBindingMap &stClassData) {
  auto kernel = std::dynamic_pointer_cast<LaunchKernelAction>(schedule.back());
  kernel->resourceStorageClassData = stClassData;
}

void VulkanRuntime::setResourceData(
    const DescriptorSetIndex desIndex, const BindingIndex bindIndex,
    const VulkanHostMemoryBuffer &hostMemBuffer) {
  // init();
  auto kernel = std::dynamic_pointer_cast<LaunchKernelAction>(schedule.back());
  kernel->resourceData[desIndex][bindIndex] = hostMemBuffer;
  kernel->resourceStorageClassData[desIndex][bindIndex] =
      spirv::StorageClass::StorageBuffer;
}

void VulkanRuntime::setEntryPoint(const char *entryPointName) {
  auto kernel = std::dynamic_pointer_cast<LaunchKernelAction>(schedule.back());
  kernel->entryPoint = entryPointName;
}

void VulkanRuntime::setResourceData(const ResourceData &resData) {
  auto kernel = std::dynamic_pointer_cast<LaunchKernelAction>(schedule.back());
  kernel->resourceData = resData;
}

void VulkanRuntime::setShaderModule(uint8_t *shader, uint32_t size) {
  auto kernel = std::dynamic_pointer_cast<LaunchKernelAction>(schedule.back());
  kernel->binary = shader;
  kernel->binarySize = size;
}

LogicalResult VulkanRuntime::mapStorageClassToDescriptorType(
    spirv::StorageClass storageClass, VkDescriptorType &descriptorType) {
  switch (storageClass) {
  case spirv::StorageClass::StorageBuffer:
    descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    break;
  case spirv::StorageClass::Uniform:
    descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    break;
  default:
    llvm::errs() << "unsupported storage class";
    return failure();
  }
  return success();
}

LogicalResult VulkanRuntime::mapStorageClassToBufferUsageFlag(
    spirv::StorageClass storageClass, VkBufferUsageFlagBits &bufferUsage) {
  switch (storageClass) {
  case spirv::StorageClass::StorageBuffer:
    bufferUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    break;
  case spirv::StorageClass::Uniform:
    bufferUsage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    break;
  default:
    llvm::errs() << "unsupported storage class";
    return failure();
  }
  return success();
}

LogicalResult VulkanRuntime::countDeviceMemorySize() {
  auto kernel = std::dynamic_pointer_cast<LaunchKernelAction>(schedule.back());
  for (const auto &resourceDataMapPair : kernel->resourceData) {
    const auto &resourceDataMap = resourceDataMapPair.second;
    for (const auto &resourceDataBindingPair : resourceDataMap) {
      if (resourceDataBindingPair.second.size) {
        memorySize += resourceDataBindingPair.second.size;
      } else {
        llvm::errs()
            << "expected buffer size greater than zero for resource data";
        return failure();
      }
    }
  }
  return success();
}

LogicalResult VulkanRuntime::initRuntime() {
  auto kernel = std::dynamic_pointer_cast<LaunchKernelAction>(schedule.back());
  if (!kernel->resourceData.size()) {
    llvm::errs() << "Vulkan runtime needs at least one resource";
    return failure();
  }
  if (!kernel->binarySize || !kernel->binary) {
    llvm::errs() << "binary shader size must be greater than zero";
    return failure();
  }
  if (failed(countDeviceMemorySize())) {
    return failure();
  }
  return success();
}

LogicalResult VulkanRuntime::destroy() {
  /*
// According to Vulkan spec:
// "To ensure that no work is active on the device, vkDeviceWaitIdle can be
// used to gate the destruction of the device. Prior to destroying a device,
// an application is responsible for destroying/freeing any Vulkan objects
// that were created using that device as the first parameter of the
// corresponding vkCreate* or vkAllocate* command."
RETURN_ON_VULKAN_ERROR(vkDeviceWaitIdle(device), "vkDeviceWaitIdle");

// Free and destroy.
vkFreeCommandBuffers(device, commandPool, commandBuffers.size(),
                 commandBuffers.data());
vkDestroyCommandPool(device, commandPool, nullptr);
vkFreeDescriptorSets(device, descriptorPool, descriptorSets.size(),
                 descriptorSets.data());
vkDestroyDescriptorPool(device, descriptorPool, nullptr);
vkDestroyPipeline(device, pipeline, nullptr);
vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
for (auto &descriptorSetLayout : descriptorSetLayouts) {
vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}
vkDestroyShaderModule(device, shaderModule, nullptr);

// For each descriptor set.
for (auto &deviceMemoryBufferMapPair : deviceMemoryBufferMap) {
auto &deviceMemoryBuffers = deviceMemoryBufferMapPair.second;
// For each descirptor binding.
for (auto &memoryBuffer : deviceMemoryBuffers) {
vkFreeMemory(device, memoryBuffer.deviceMemory, nullptr);
vkDestroyBuffer(device, memoryBuffer.buffer, nullptr);
}
}
*/
  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
  return success();
}
LogicalResult VulkanRuntime::init() {
  createInstance();
  createDevice();

  // Get working queue.
  vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

  return success();
}

LogicalResult VulkanRuntime::createAction() {
  schedule.push_back(std::make_shared<LaunchKernelAction>());
  return success();
}

LogicalResult VulkanRuntime::run() {
  initRuntime();
  // Create logical device, shader module and memory buffers.
  if (failed(createMemoryBuffers()) || failed(createShaderModule())) {
    return failure();
  }

  // Descriptor bindings divided into sets. Each descriptor binding
  // must have a layout binding attached into a descriptor set layout.
  // Each layout set must be binded into a pipeline layout.
  initDescriptorSetLayoutBindingMap();
  if (failed(createDescriptorSetLayout()) || failed(createPipelineLayout()) ||
      // Each descriptor set must be allocated from a descriptor pool.
      failed(createComputePipeline()) || failed(createDescriptorPool()) ||
      failed(allocateDescriptorSets()) || failed(setWriteDescriptors()) ||
      // Create command buffer.
      failed(createCommandPool())) {
    return failure();
  }

  if (failed(createSchedule())) {
    return failure();
  }

  return success();
}

LogicalResult VulkanRuntime::submitBuffer() {
  // Submit command buffer into the queue.
  if (failed(submitCommandBuffersToQueue()))
    return failure();

  RETURN_ON_VULKAN_ERROR(vkQueueWaitIdle(queue), "vkQueueWaitIdle");

  updateHostMemoryBuffers();
  return success();
}

LogicalResult VulkanRuntime::createInstance() {
  RETURN_ON_VULKAN_ERROR(volkInitialize(), "volkInitialize");

  VkApplicationInfo applicationInfo = {};
  applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  applicationInfo.pNext = nullptr;
  applicationInfo.pApplicationName = "MLIR Vulkan runtime";
  applicationInfo.applicationVersion = 0;
  applicationInfo.pEngineName = "mlir";
  applicationInfo.engineVersion = 0;
  applicationInfo.apiVersion = VK_MAKE_VERSION(1, 0, 0);

  VkInstanceCreateInfo instanceCreateInfo = {};
  instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instanceCreateInfo.pNext = nullptr;
  instanceCreateInfo.flags = 0;
  instanceCreateInfo.pApplicationInfo = &applicationInfo;
  instanceCreateInfo.enabledLayerCount = 0;
  instanceCreateInfo.ppEnabledLayerNames = 0;
  instanceCreateInfo.enabledExtensionCount = 0;
  instanceCreateInfo.ppEnabledExtensionNames = 0;

  RETURN_ON_VULKAN_ERROR(vkCreateInstance(&instanceCreateInfo, 0, &instance),
                         "vkCreateInstance");

  volkLoadInstance(instance);

  return success();
}

LogicalResult VulkanRuntime::createDevice() {
  uint32_t physicalDeviceCount = 0;
  RETURN_ON_VULKAN_ERROR(
      vkEnumeratePhysicalDevices(instance, &physicalDeviceCount, 0),
      "vkEnumeratePhysicalDevices");

  SmallVector<VkPhysicalDevice, 1> physicalDevices(physicalDeviceCount);
  RETURN_ON_VULKAN_ERROR(vkEnumeratePhysicalDevices(instance,
                                                    &physicalDeviceCount,
                                                    physicalDevices.data()),
                         "vkEnumeratePhysicalDevices");

  RETURN_ON_VULKAN_ERROR(physicalDeviceCount ? VK_SUCCESS : VK_INCOMPLETE,
                         "physicalDeviceCount");

  // TODO(denis0x0D): find the best device.
  const auto &physicalDevice = physicalDevices.front();
  getBestComputeQueue(physicalDevice);

  const float queuePrioritory = 1.0f;
  VkDeviceQueueCreateInfo deviceQueueCreateInfo = {};
  deviceQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  deviceQueueCreateInfo.pNext = nullptr;
  deviceQueueCreateInfo.flags = 0;
  deviceQueueCreateInfo.queueFamilyIndex = queueFamilyIndex;
  deviceQueueCreateInfo.queueCount = 1;
  deviceQueueCreateInfo.pQueuePriorities = &queuePrioritory;

  // Structure specifying parameters of a newly created device.
  VkDeviceCreateInfo deviceCreateInfo = {};
  deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceCreateInfo.pNext = nullptr;
  deviceCreateInfo.flags = 0;
  deviceCreateInfo.queueCreateInfoCount = 1;
  deviceCreateInfo.pQueueCreateInfos = &deviceQueueCreateInfo;
  deviceCreateInfo.enabledLayerCount = 0;
  deviceCreateInfo.ppEnabledLayerNames = nullptr;
  deviceCreateInfo.enabledExtensionCount = 0;
  deviceCreateInfo.ppEnabledExtensionNames = nullptr;
  deviceCreateInfo.pEnabledFeatures = nullptr;

  RETURN_ON_VULKAN_ERROR(
      vkCreateDevice(physicalDevice, &deviceCreateInfo, 0, &device),
      "vkCreateDevice");

  VkPhysicalDeviceMemoryProperties properties = {};
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &properties);

  // Try to find memory type with following properties:
  // VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT bit specifies that memory allocated
  // with this type can be mapped for host access using vkMapMemory;
  // VK_MEMORY_PROPERTY_HOST_COHERENT_BIT bit specifies that the host cache
  // management commands vkFlushMappedMemoryRanges and
  // vkInvalidateMappedMemoryRanges are not needed to flush host writes to the
  // device or make device writes visible to the host, respectively.
  for (uint32_t i = 0, e = properties.memoryTypeCount; i < e; ++i) {
    if ((VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT &
         properties.memoryTypes[i].propertyFlags) &&
        (VK_MEMORY_PROPERTY_HOST_COHERENT_BIT &
         properties.memoryTypes[i].propertyFlags) &&
        (memorySize <=
         properties.memoryHeaps[properties.memoryTypes[i].heapIndex].size)) {
      memoryTypeIndex = i;
      break;
    }
  }

  RETURN_ON_VULKAN_ERROR(memoryTypeIndex == VK_MAX_MEMORY_TYPES ? VK_INCOMPLETE
                                                                : VK_SUCCESS,
                         "invalid memoryTypeIndex");
  return success();
}

LogicalResult
VulkanRuntime::getBestComputeQueue(const VkPhysicalDevice &physicalDevice) {
  uint32_t queueFamilyPropertiesCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice,
                                           &queueFamilyPropertiesCount, 0);
  SmallVector<VkQueueFamilyProperties, 1> queueFamilyProperties(
      queueFamilyPropertiesCount);

  vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice,
                                           &queueFamilyPropertiesCount,
                                           queueFamilyProperties.data());

  // VK_QUEUE_COMPUTE_BIT specifies that queues in this queue family support
  // compute operations.
  for (uint32_t i = 0; i < queueFamilyPropertiesCount; ++i) {
    const VkQueueFlags maskedFlags =
        (~(VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT) &
         queueFamilyProperties[i].queueFlags);

    if (!(VK_QUEUE_GRAPHICS_BIT & maskedFlags) &&
        (VK_QUEUE_COMPUTE_BIT & maskedFlags)) {
      queueFamilyIndex = i;
      return success();
    }
  }

  for (uint32_t i = 0; i < queueFamilyPropertiesCount; ++i) {
    const VkQueueFlags maskedFlags =
        (~(VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT) &
         queueFamilyProperties[i].queueFlags);

    if (VK_QUEUE_COMPUTE_BIT & maskedFlags) {
      queueFamilyIndex = i;
      return success();
    }
  }

  llvm::errs() << "cannot find valid queue";
  return failure();
}

LogicalResult VulkanRuntime::createMemoryBuffers() {
  auto kernel = std::dynamic_pointer_cast<LaunchKernelAction>(schedule.back());
  // For each descriptor set.
  for (const auto &resourceDataMapPair : kernel->resourceData) {
    llvm::SmallVector<VulkanDeviceMemoryBuffer, 1> deviceMemoryBuffers;
    const auto descriptorSetIndex = resourceDataMapPair.first;
    const auto &resourceDataMap = resourceDataMapPair.second;

    // For each descriptor binding.
    for (const auto &resourceDataBindingPair : resourceDataMap) {
      // Create device memory buffer.
      VulkanDeviceMemoryBuffer memoryBuffer;
      memoryBuffer.bindingIndex = resourceDataBindingPair.first;
      VkDescriptorType descriptorType = {};
      VkBufferUsageFlagBits bufferUsage = {};

      // Check that descriptor set has storage class map.
      const auto resourceStorageClassMapIt =
          kernel->resourceStorageClassData.find(descriptorSetIndex);
      if (resourceStorageClassMapIt == kernel->resourceStorageClassData.end()) {
        llvm::errs()
            << "cannot find storge class for resource in descriptor set: "
            << descriptorSetIndex;
        return failure();
      }

      // Check that specific descriptor binding has storage class.
      const auto &resourceStorageClassMap = resourceStorageClassMapIt->second;
      const auto resourceStorageClassIt =
          resourceStorageClassMap.find(resourceDataBindingPair.first);
      if (resourceStorageClassIt == resourceStorageClassMap.end()) {
        llvm::errs()
            << "cannot find storage class for resource with descriptor index: "
            << resourceDataBindingPair.first;
        return failure();
      }

      const auto resourceStorageClassBinding = resourceStorageClassIt->second;
      if (failed(mapStorageClassToDescriptorType(resourceStorageClassBinding,
                                                 descriptorType)) ||
          failed(mapStorageClassToBufferUsageFlag(resourceStorageClassBinding,
                                                  bufferUsage))) {
        llvm::errs() << "storage class for resource with descriptor binding: "
                     << resourceDataBindingPair.first
                     << " in the descriptor set: " << descriptorSetIndex
                     << " is not supported ";
        return failure();
      }

      // Set descriptor type for the specific device memory buffer.
      memoryBuffer.descriptorType = descriptorType;
      const auto bufferSize = resourceDataBindingPair.second.size;

      // Specify memory allocation info.
      VkMemoryAllocateInfo memoryAllocateInfo = {};
      memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      memoryAllocateInfo.pNext = nullptr;
      memoryAllocateInfo.allocationSize = bufferSize;
      memoryAllocateInfo.memoryTypeIndex = memoryTypeIndex;

      // Allocate device memory.
      RETURN_ON_VULKAN_ERROR(vkAllocateMemory(device, &memoryAllocateInfo, 0,
                                              &memoryBuffer.deviceMemory),
                             "vkAllocateMemory");
      void *payload;
      RETURN_ON_VULKAN_ERROR(vkMapMemory(device, memoryBuffer.deviceMemory, 0,
                                         bufferSize, 0,
                                         reinterpret_cast<void **>(&payload)),
                             "vkMapMemory");

      // Copy host memory into the mapped area.
      std::memcpy(payload, resourceDataBindingPair.second.ptr, bufferSize);
      vkUnmapMemory(device, memoryBuffer.deviceMemory);

      VkBufferCreateInfo bufferCreateInfo = {};
      bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
      bufferCreateInfo.pNext = nullptr;
      bufferCreateInfo.flags = 0;
      bufferCreateInfo.size = bufferSize;
      bufferCreateInfo.usage = bufferUsage;
      bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      bufferCreateInfo.queueFamilyIndexCount = 1;
      bufferCreateInfo.pQueueFamilyIndices = &queueFamilyIndex;
      RETURN_ON_VULKAN_ERROR(
          vkCreateBuffer(device, &bufferCreateInfo, 0, &memoryBuffer.buffer),
          "vkCreateBuffer");

      // Bind buffer and device memory.
      RETURN_ON_VULKAN_ERROR(vkBindBufferMemory(device, memoryBuffer.buffer,
                                                memoryBuffer.deviceMemory, 0),
                             "vkBindBufferMemory");

      // Update buffer info.
      memoryBuffer.bufferInfo.buffer = memoryBuffer.buffer;
      memoryBuffer.bufferInfo.offset = 0;
      memoryBuffer.bufferInfo.range = VK_WHOLE_SIZE;
      deviceMemoryBuffers.push_back(memoryBuffer);
    }

    // Associate device memory buffers with a descriptor set.
    kernel->deviceMemoryBufferMap[descriptorSetIndex] = deviceMemoryBuffers;
  }
  return success();
}

LogicalResult VulkanRuntime::createShaderModule() {
  auto kernel = std::dynamic_pointer_cast<LaunchKernelAction>(schedule.back());
  VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
  shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shaderModuleCreateInfo.pNext = nullptr;
  shaderModuleCreateInfo.flags = 0;
  // Set size in bytes.
  shaderModuleCreateInfo.codeSize = kernel->binarySize;
  // Set pointer to the binary shader.
  shaderModuleCreateInfo.pCode = reinterpret_cast<uint32_t *>(kernel->binary);
  RETURN_ON_VULKAN_ERROR(vkCreateShaderModule(device, &shaderModuleCreateInfo,
                                              0, &kernel->shaderModule),
                         "vkCreateShaderModule");
  return success();
}

void VulkanRuntime::initDescriptorSetLayoutBindingMap() {
  auto kernel = std::dynamic_pointer_cast<LaunchKernelAction>(schedule.back());
  for (const auto &deviceMemoryBufferMapPair : kernel->deviceMemoryBufferMap) {
    SmallVector<VkDescriptorSetLayoutBinding, 1> descriptorSetLayoutBindings;
    const auto &deviceMemoryBuffers = deviceMemoryBufferMapPair.second;
    const auto descriptorSetIndex = deviceMemoryBufferMapPair.first;

    // Create a layout binding for each descriptor.
    for (const auto &memBuffer : deviceMemoryBuffers) {
      VkDescriptorSetLayoutBinding descriptorSetLayoutBinding = {};
      descriptorSetLayoutBinding.binding = memBuffer.bindingIndex;
      descriptorSetLayoutBinding.descriptorType = memBuffer.descriptorType;
      descriptorSetLayoutBinding.descriptorCount = 1;
      descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
      descriptorSetLayoutBinding.pImmutableSamplers = 0;
      descriptorSetLayoutBindings.push_back(descriptorSetLayoutBinding);
    }
    kernel->descriptorSetLayoutBindingMap[descriptorSetIndex] =
        descriptorSetLayoutBindings;
  }
}

LogicalResult VulkanRuntime::createDescriptorSetLayout() {
  auto kernel = std::dynamic_pointer_cast<LaunchKernelAction>(schedule.back());
  for (const auto &deviceMemoryBufferMapPair : kernel->deviceMemoryBufferMap) {
    const auto descriptorSetIndex = deviceMemoryBufferMapPair.first;
    const auto &deviceMemoryBuffers = deviceMemoryBufferMapPair.second;
    // Each descriptor in a descriptor set must be the same type.
    VkDescriptorType descriptorType =
        deviceMemoryBuffers.front().descriptorType;
    const uint32_t descriptorSize = deviceMemoryBuffers.size();
    const auto descriptorSetLayoutBindingIt =
        kernel->descriptorSetLayoutBindingMap.find(descriptorSetIndex);

    if (descriptorSetLayoutBindingIt ==
        kernel->descriptorSetLayoutBindingMap.end()) {
      llvm::errs() << "cannot find layout bindings for the set with number: "
                   << descriptorSetIndex;
      return failure();
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
    RETURN_ON_VULKAN_ERROR(
        vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, 0,
                                    &descriptorSetLayout),
        "vkCreateDescriptorSetLayout");

    kernel->descriptorSetLayouts.push_back(descriptorSetLayout);
    kernel->descriptorSetInfoPool.push_back(
        {descriptorSetIndex, descriptorSize, descriptorType});
  }
  return success();
}

LogicalResult VulkanRuntime::createPipelineLayout() {
  auto kernel = std::dynamic_pointer_cast<LaunchKernelAction>(schedule.back());
  // Associate descriptor sets with a pipeline layout.
  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
  pipelineLayoutCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutCreateInfo.pNext = nullptr;
  pipelineLayoutCreateInfo.flags = 0;
  pipelineLayoutCreateInfo.setLayoutCount = kernel->descriptorSetLayouts.size();
  pipelineLayoutCreateInfo.pSetLayouts = kernel->descriptorSetLayouts.data();
  pipelineLayoutCreateInfo.pushConstantRangeCount = 0;
  pipelineLayoutCreateInfo.pPushConstantRanges = 0;
  RETURN_ON_VULKAN_ERROR(vkCreatePipelineLayout(device,
                                                &pipelineLayoutCreateInfo, 0,
                                                &kernel->pipelineLayout),
                         "vkCreatePipelineLayout");
  return success();
}

LogicalResult VulkanRuntime::createComputePipeline() {
  auto kernel = std::dynamic_pointer_cast<LaunchKernelAction>(schedule.back());
  VkPipelineShaderStageCreateInfo stageInfo = {};
  stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stageInfo.pNext = nullptr;
  stageInfo.flags = 0;
  stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  stageInfo.module = kernel->shaderModule;
  // Set entry point.
  stageInfo.pName = kernel->entryPoint;
  stageInfo.pSpecializationInfo = 0;

  VkComputePipelineCreateInfo computePipelineCreateInfo = {};
  computePipelineCreateInfo.sType =
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  computePipelineCreateInfo.pNext = nullptr;
  computePipelineCreateInfo.flags = 0;
  computePipelineCreateInfo.stage = stageInfo;
  computePipelineCreateInfo.layout = kernel->pipelineLayout;
  computePipelineCreateInfo.basePipelineHandle = 0;
  computePipelineCreateInfo.basePipelineIndex = 0;
  RETURN_ON_VULKAN_ERROR(vkCreateComputePipelines(device, 0, 1,
                                                  &computePipelineCreateInfo, 0,
                                                  &kernel->pipeline),
                         "vkCreateComputePipelines");
  return success();
}

LogicalResult VulkanRuntime::createDescriptorPool() {
  auto kernel = std::dynamic_pointer_cast<LaunchKernelAction>(schedule.back());
  SmallVector<VkDescriptorPoolSize, 1> descriptorPoolSizes;
  for (const auto &descriptorSetInfo : kernel->descriptorSetInfoPool) {
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
  RETURN_ON_VULKAN_ERROR(vkCreateDescriptorPool(device,
                                                &descriptorPoolCreateInfo, 0,
                                                &kernel->descriptorPool),
                         "vkCreateDescriptorPool");
  return success();
}

LogicalResult VulkanRuntime::allocateDescriptorSets() {
  auto kernel = std::dynamic_pointer_cast<LaunchKernelAction>(schedule.back());
  VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
  // Size of desciptor sets and descriptor layout sets is the same.
  kernel->descriptorSets.resize(kernel->descriptorSetLayouts.size());
  descriptorSetAllocateInfo.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  descriptorSetAllocateInfo.pNext = nullptr;
  descriptorSetAllocateInfo.descriptorPool = kernel->descriptorPool;
  descriptorSetAllocateInfo.descriptorSetCount =
      kernel->descriptorSetLayouts.size();
  descriptorSetAllocateInfo.pSetLayouts = kernel->descriptorSetLayouts.data();
  RETURN_ON_VULKAN_ERROR(
      vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo,
                               kernel->descriptorSets.data()),
      "vkAllocateDescriptorSets");
  return success();
}

LogicalResult VulkanRuntime::setWriteDescriptors() {
  auto kernel = std::dynamic_pointer_cast<LaunchKernelAction>(schedule.back());
  if (kernel->descriptorSets.size() != kernel->descriptorSetInfoPool.size()) {
    llvm::errs() << "Each descriptor set must have descriptor set information";
    return failure();
  }
  // For each descriptor set.
  auto descriptorSetIt = kernel->descriptorSets.begin();
  // Each descriptor set is associated with descriptor set info.
  for (const auto &descriptorSetInfo : kernel->descriptorSetInfoPool) {
    // For each device memory buffer in the descriptor set.
    const auto &deviceMemoryBuffers =
        kernel->deviceMemoryBufferMap[descriptorSetInfo.descriptorSet];
    for (const auto &memoryBuffer : deviceMemoryBuffers) {
      // Structure describing descriptor sets to write to.
      VkWriteDescriptorSet wSet = {};
      wSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      wSet.pNext = nullptr;
      // Descirptor set.
      wSet.dstSet = *descriptorSetIt;
      wSet.dstBinding = memoryBuffer.bindingIndex;
      wSet.dstArrayElement = 0;
      wSet.descriptorCount = 1;
      wSet.descriptorType = memoryBuffer.descriptorType;
      wSet.pImageInfo = nullptr;
      wSet.pBufferInfo = &memoryBuffer.bufferInfo;
      wSet.pTexelBufferView = nullptr;
      vkUpdateDescriptorSets(device, 1, &wSet, 0, nullptr);
    }
    // Increment descriptor set iterator.
    ++descriptorSetIt;
  }
  return success();
}

LogicalResult VulkanRuntime::createCommandPool() {
  auto kernel = std::dynamic_pointer_cast<LaunchKernelAction>(schedule.back());
  VkCommandPoolCreateInfo commandPoolCreateInfo = {};
  commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  commandPoolCreateInfo.pNext = nullptr;
  commandPoolCreateInfo.flags = 0;
  commandPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;
  RETURN_ON_VULKAN_ERROR(vkCreateCommandPool(device, &commandPoolCreateInfo, 0,
                                             &kernel->commandPool),
                         "vkCreateCommandPool");
  return success();
}

LogicalResult VulkanRuntime::createComputeCommandBuffer() {
  /*
VkCommandBufferAllocateInfo commandBufferAllocateInfo = {};
VkCommandBuffer commandBuffer;
commandBufferAllocateInfo.sType =
VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
commandBufferAllocateInfo.pNext = nullptr;
commandBufferAllocateInfo.commandPool = commandPool;
commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
commandBufferAllocateInfo.commandBufferCount = 1;
RETURN_ON_VULKAN_ERROR(vkAllocateCommandBuffers(device,
                                            &commandBufferAllocateInfo,
                                            &commandBuffer),
                   "vkAllocateCommandBuffers");

VkCommandBufferBeginInfo commandBufferBeginInfo = {};
commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
commandBufferBeginInfo.pNext = nullptr;
commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
commandBufferBeginInfo.pInheritanceInfo = nullptr;

// Commands begin.
RETURN_ON_VULKAN_ERROR(
vkBeginCommandBuffer(commandBuffer, &commandBufferBeginInfo),
"vkBeginCommandBuffer");

// Commands.
vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    pipelineLayout, 0, descriptorSets.size(),
                    descriptorSets.data(), 0, 0);
vkCmdDispatch(commandBuffer, numWorkGroups.x, numWorkGroups.y,
          numWorkGroups.z);

// Commands end.
RETURN_ON_VULKAN_ERROR(vkEndCommandBuffer(commandBuffer),
                   "vkEndCommandBuffer");

commandBuffers.push_back(commandBuffer);
*/
  return success();
}

LogicalResult VulkanRuntime::submitCommandBuffersToQueue() {
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
  RETURN_ON_VULKAN_ERROR(vkQueueSubmit(queue, 1, &submitInfo, 0),
                         "vkQueueSubmit");
  return success();
}

LogicalResult VulkanRuntime::updateHostMemoryBuffers() {
  auto kernel = std::dynamic_pointer_cast<LaunchKernelAction>(schedule.back());
  // For each descriptor set.
  for (auto &resourceDataMapPair : kernel->resourceData) {
    auto &resourceDataMap = resourceDataMapPair.second;
    auto &deviceMemoryBuffers =
        kernel->deviceMemoryBufferMap[resourceDataMapPair.first];
    // For each device memory buffer in the set.
    for (auto &deviceMemoryBuffer : deviceMemoryBuffers) {
      if (resourceDataMap.count(deviceMemoryBuffer.bindingIndex)) {
        void *payload;
        auto &hostMemoryBuffer =
            resourceDataMap[deviceMemoryBuffer.bindingIndex];
        RETURN_ON_VULKAN_ERROR(vkMapMemory(device,
                                           deviceMemoryBuffer.deviceMemory, 0,
                                           hostMemoryBuffer.size, 0,
                                           reinterpret_cast<void **>(&payload)),
                               "vkMapMemory");
        std::memcpy(hostMemoryBuffer.ptr, payload, hostMemoryBuffer.size);
        vkUnmapMemory(device, deviceMemoryBuffer.deviceMemory);
      }
    }
  }
  return success();
}

LogicalResult VulkanRuntime::createSchedule() {
  auto kernel = std::dynamic_pointer_cast<LaunchKernelAction>(schedule.back());
  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.pNext = nullptr;
  allocInfo.commandPool = kernel->commandPool;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer;
  RETURN_ON_VULKAN_ERROR(
      vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer),
      "vkAllocateCommandBuffers");

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.pNext = nullptr;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  beginInfo.pInheritanceInfo = nullptr;

  RETURN_ON_VULKAN_ERROR(vkBeginCommandBuffer(commandBuffer, &beginInfo),
                         "vkBeginCommandBuffer");

  for (const auto &action : schedule) {
    if (auto kernel = std::dynamic_pointer_cast<LaunchKernelAction>(action)) {
      if (kernel->deps.size()) {
        vkCmdPipelineBarrier(
            commandBuffer,
            /*srcStageMask=*/VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            /*dstStageMask=*/VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            /*dependencyFlags=*/0, /*memoryBarrierCount=*/0,
            /*pMemoryBarriers=*/nullptr,
            /*bufferMemoryBarrierCount=*/kernel->deps.size(),
            /*pBufferMemoryBarriers=*/kernel->deps.data(),
            /*imageMemoryBarrierCount=*/0,
            /*pImageMemoryBarriers=*/nullptr);
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
    }

    if (auto xfer = std::dynamic_pointer_cast<MemoryTransferAction>(action)) {
      vkCmdCopyBuffer(commandBuffer,
                      /*srcBuffer=*/xfer->src,
                      /*dstBuffer=*/xfer->dst,
                      /*regionCount=*/xfer->regions.size(),
                      /*pRegions=*/xfer->regions.data());
    }
  }

  RETURN_ON_VULKAN_ERROR(vkEndCommandBuffer(commandBuffer),
                         "vkEndCommandBuffer");

  commandBuffers.push_back(commandBuffer);
  return success();
}
