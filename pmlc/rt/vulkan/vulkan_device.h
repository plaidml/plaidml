// Vulkan device interface, originally from the LLVM project, and subsequently
// modified by Intel Corporation.
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
#include <set>
#include <string>
#include <vector>

#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Serialization.h"
#include "mlir/IR/Module.h"
#include "mlir/Support/LLVM.h"
#include "pmlc/rt/runtime.h"
#include "pmlc/rt/vulkan/vulkan_state.h"
#include "llvm/Support/ToolOutputFile.h"

#include "volk.h" // NOLINT[build/include_subdir]

namespace pmlc::rt::vulkan {

/// Vulkan device abstraction.
class VulkanDevice final : public pmlc::rt::Device,
                           public std::enable_shared_from_this<VulkanDevice> {
public:
  VulkanDevice(const VkPhysicalDevice &physicalDevice,
               std::shared_ptr<VulkanState> state);
  ~VulkanDevice();
  VulkanDevice(const VulkanDevice &) = delete;
  VulkanDevice &operator=(const VulkanDevice &) = delete;

  std::unique_ptr<Executable>
  compile(const std::shared_ptr<pmlc::compiler::Program> &program) final;

  const VkDevice &getDevice() const { return device; }
  const VkQueue &getQueue() const { return queue; }
  uint32_t getQueueFamilyIndex() const { return queueFamilyIndex; }
  uint32_t getMemoryTypeIndex() const { return memoryTypeIndex; }
  float getTimestampPeriod() const { return timestampPeriod; }
  uint32_t getTimestampValidBits() const { return timestampValidBits; }

  bool isExtensionSupported(const std::string &extension_name);

private:
  void getBestComputeQueue(const VkPhysicalDevice &physicalDevice);
  void getExtensions(const VkPhysicalDevice &physicalDevice);

  // Accessor for the Vulkan runtime state.
  std::shared_ptr<VulkanState> state;

  // Performance monitoring
  float timestampPeriod{0.0};
  uint32_t timestampValidBits{0};

  //===--------------------------------------------------------------------===//
  // Vulkan objects.
  //===--------------------------------------------------------------------===//
  VkDevice device;
  VkQueue queue;

  //===--------------------------------------------------------------------===//
  // Vulkan memory context.
  //===--------------------------------------------------------------------===//
  uint32_t queueFamilyIndex{0};
  uint32_t memoryTypeIndex{VK_MAX_MEMORY_TYPES};

  std::set<std::string> extensionList;
};

} // namespace pmlc::rt::vulkan
