#pragma once

#include <dlfcn.h>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#define EXPORT_VULKAN_FUNCTION(instance, function) \
  function = (PFN_##function)vkGetInstanceProcAddr(instance, #function);

namespace pmlc::vulkan {

VkInstance* _pInstance = nullptr;

PFN_vkDestroyCommandPool vkDestroyCommandPool;
PFN_vkFreeDescriptorSets vkFreeDescriptorSets;
PFN_vkDestroyPipeline vkDestroyPipeline;
PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout;
PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout;
PFN_vkDestroyShaderModule vkDestroyShaderModule;
PFN_vkFreeCommandBuffers vkFreeCommandBuffers;
PFN_vkBindBufferMemory vkBindBufferMemory;
PFN_vkMapMemory vkMapMemory;
PFN_vkDestroyInstance vkDestroyInstance;
PFN_vkCreateShaderModule vkCreateShaderModule;
PFN_vkDestroyDescriptorPool vkDestroyDescriptorPool;
PFN_vkFreeMemory vkFreeMemory;
PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout;
PFN_vkAllocateDescriptorSets vkAllocateDescriptorSets;
PFN_vkCreateDevice vkCreateDevice;
PFN_vkCreateDescriptorPool vkCreateDescriptorPool;
PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties;
PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets;
PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers;
PFN_vkBeginCommandBuffer vkBeginCommandBuffer;
PFN_vkDestroyDevice vkDestroyDevice;
PFN_vkCreatePipelineLayout vkCreatePipelineLayout;
PFN_vkCreateComputePipelines vkCreateComputePipelines;
PFN_vkEndCommandBuffer vkEndCommandBuffer;
PFN_vkCreateCommandPool vkCreateCommandPool;
PFN_vkCmdDispatch vkCmdDispatch;
PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties;
PFN_vkAllocateMemory vkAllocateMemory;
PFN_vkDestroyBuffer vkDestroyBuffer;
PFN_vkUnmapMemory vkUnmapMemory;
PFN_vkUpdateDescriptorSets vkUpdateDescriptorSets;
PFN_vkCreateBuffer vkCreateBuffer;
PFN_vkGetDeviceQueue vkGetDeviceQueue;
PFN_vkQueueWaitIdle vkQueueWaitIdle;
PFN_vkQueueSubmit vkQueueSubmit;
PFN_vkCmdBindPipeline vkCmdBindPipeline;
PFN_vkDeviceWaitIdle vkDeviceWaitIdle;
PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices;
PFN_vkEnumerateInstanceVersion vkEnumerateInstanceVersion;
PFN_vkEnumerateInstanceExtensionProperties vkEnumerateInstanceExtensionProperties;
PFN_vkEnumerateInstanceLayerProperties vkEnumerateInstanceLayerProperties;

PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr =
    reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(dlopen("libvulkan.so", RTLD_NOW), "vkGetInstanceProcAddr"));

void exportAllFunctions() {
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkDestroyCommandPool)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkFreeDescriptorSets)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkDestroyPipeline)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkDestroyPipelineLayout)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkDestroyDescriptorSetLayout)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkDestroyShaderModule)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkFreeCommandBuffers)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkBindBufferMemory)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkMapMemory)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkDestroyInstance)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkCreateShaderModule)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkDestroyDescriptorPool)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkFreeMemory)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkCreateDescriptorSetLayout)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkAllocateDescriptorSets)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkCreateDevice)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkCreateDescriptorPool)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkGetPhysicalDeviceMemoryProperties)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkCmdBindDescriptorSets)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkAllocateCommandBuffers)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkBeginCommandBuffer)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkDestroyDevice)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkCreatePipelineLayout)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkCreateComputePipelines)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkEndCommandBuffer)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkCreateCommandPool)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkCmdDispatch)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkGetPhysicalDeviceQueueFamilyProperties)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkAllocateMemory)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkDestroyBuffer)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkUnmapMemory)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkUpdateDescriptorSets)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkCreateBuffer)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkGetDeviceQueue)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkQueueWaitIdle)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkQueueSubmit)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkCmdBindPipeline)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkDeviceWaitIdle)
  EXPORT_VULKAN_FUNCTION(*_pInstance, vkEnumeratePhysicalDevices)
  EXPORT_VULKAN_FUNCTION(NULL, vkEnumerateInstanceVersion)
  EXPORT_VULKAN_FUNCTION(NULL, vkEnumerateInstanceExtensionProperties)
  EXPORT_VULKAN_FUNCTION(NULL, vkEnumerateInstanceLayerProperties)
}

VkResult vkCreateInstance(const VkInstanceCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator,
                          VkInstance* pInstance) {
  auto _vkCreateInstance = (PFN_vkCreateInstance)vkGetInstanceProcAddr(NULL, "vkCreateInstance");
  auto result = _vkCreateInstance(pCreateInfo, pAllocator, pInstance);
  _pInstance = pInstance;
  exportAllFunctions();
  return result;
}
}  // namespace pmlc::vulkan
