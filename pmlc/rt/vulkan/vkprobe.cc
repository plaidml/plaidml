// Copyright 2020 Intel Corporation

#include <iostream>

#include "volk.h" // NOLINT[build/include_subdir]

int main(int argc, char **argv) {
  auto result = volkInitialize();
  if (result != VK_SUCCESS) {
    std::cerr << "volkInitialize failed" << std::endl;
    return 1;
  }

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

  VkInstance instance;
  result = vkCreateInstance(&instanceCreateInfo, 0, &instance);
  if (result != VK_SUCCESS) {
    std::cerr << "vkCreateInstance failed" << std::endl;
    return 1;
  }

  return 0;
}
