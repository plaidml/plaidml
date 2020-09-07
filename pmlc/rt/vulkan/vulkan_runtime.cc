// Copyright 2020, Intel Corporation

#include <stdexcept>

#include "pmlc/rt/runtime_registry.h"
#include "pmlc/rt/vulkan/vulkan_device.h"
#include "pmlc/rt/vulkan/vulkan_error.h"
#include "pmlc/rt/vulkan/vulkan_state.h"

namespace pmlc::rt::vulkan {

class VulkanRuntime final : public pmlc::rt::Runtime {
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
  std::shared_ptr<pmlc::rt::Device> device(std::size_t idx) {
    return devices.at(idx);
  }

private:
  std::vector<std::shared_ptr<VulkanDevice>> devices;
};

pmlc::rt::RuntimeRegistration<VulkanRuntime> reg{"vulkan"};

} // namespace pmlc::rt::vulkan
