// Copyright 2020, Intel Corporation

#include "pmlc/rt/vulkan/vulkan_state.h"

#include "pmlc/rt/vulkan/vulkan_error.h"

namespace pmlc::rt::vulkan {

VulkanState::VulkanState() {
  throwOnVulkanError(volkInitialize(), "volkInitialize");

  VkApplicationInfo applicationInfo = {};
  applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  applicationInfo.pNext = nullptr;
  applicationInfo.pApplicationName = "MLIR Vulkan Runtime";
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

  throwOnVulkanError(vkCreateInstance(&instanceCreateInfo, 0, &instance),
                     "vkCreateInstance");
  volkLoadInstance(instance);
}

VulkanState::~VulkanState() { vkDestroyInstance(instance, nullptr); }

} // namespace pmlc::rt::vulkan
