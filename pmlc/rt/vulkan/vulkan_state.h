// Copyright 2020, Intel Corporation

#pragma once

#include "volk.h" // NOLINT[build/include_subdir]

namespace pmlc::rt::vulkan {

// Encapsulates the state of the Vulkan runtime that needs to be available to
// the individual Vulkan devices.
class VulkanState final {
public:
  VulkanState();
  ~VulkanState();

  VkInstance instance;
};

} // namespace pmlc::rt::vulkan
