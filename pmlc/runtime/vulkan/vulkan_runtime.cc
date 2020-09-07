// Copyright 2020, Intel Corporation

#include <stdexcept>

#include "pmlc/runtime/runtime_registry.h"
#include "pmlc/runtime/vulkan/vulkan_device.h"
#include "pmlc/runtime/vulkan/vulkan_error.h"
#include "pmlc/runtime/vulkan/vulkan_state.h"

namespace pmlc::runtime::vulkan {

class VulkanRuntime final : public pmlc::runtime::Runtime {
public:
  VulkanRuntime() {
    auto state = std::make_shared<VulkanState>();

    uint32_t physicalDeviceCount = 0;
    throwOnVulkanError(vkEnumeratePhysicalDevices(
                           state->instance, &physicalDeviceCount, nullptr),
                       "vkEnumeratePhysicalDevices");

    llvm::SmallVector<VkPhysicalDevice, 1> physicalDevices(physicalDeviceCount);
    throwOnVulkanError(vkEnumeratePhysicalDevices(state->instance,
                                                  &physicalDeviceCount,
                                                  physicalDevices.data()),
                       "vkEnumeratePhysicalDevices");

    for (const auto &physicalDevice : physicalDevices) {
      devices.emplace_back(
          std::make_shared<VulkanDevice>(physicalDevice, state));
    }
  }

  std::size_t deviceCount() const noexcept final { return devices.size(); }
  std::shared_ptr<pmlc::runtime::Device> device(std::size_t idx) {
    return devices.at(idx);
  }

private:
  std::vector<std::shared_ptr<VulkanDevice>> devices;
};

pmlc::runtime::RuntimeRegistration<VulkanRuntime> reg{"vulkan"};

} // namespace pmlc::runtime::vulkan
