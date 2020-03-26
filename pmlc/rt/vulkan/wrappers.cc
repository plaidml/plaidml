//===- vulkan-runtime-wrappers.cpp - MLIR Vulkan runner wrapper library ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements C runtime wrappers around the VulkanRuntime.
//
//===----------------------------------------------------------------------===//

#include <mutex>
#include <numeric>

#include "llvm/Support/raw_ostream.h"

#include "pmlc/rt/vulkan/vulkan_runtime.h"

#ifdef _WIN32
#ifndef VULKAN_RT_EXPORT
#ifdef VULKAN_RT_BUILD
/* We are building this library */
#define VULKAN_RT_EXPORT __declspec(dllexport)
#else
/* We are using this library */
#define VULKAN_RT_EXPORT __declspec(dllimport)
#endif // VULKAN_RT_BUILD
#endif // VULKAN_RT_EXPORT
#else
#define VULKAN_RT_EXPORT
#endif // _WIN32

namespace {

class VulkanRuntimeManager {
public:
  VulkanRuntimeManager() = default;
  VulkanRuntimeManager(const VulkanRuntimeManager &) = delete;
  VulkanRuntimeManager operator=(const VulkanRuntimeManager &) = delete;
  ~VulkanRuntimeManager() = default;

  void setResourceData(DescriptorSetIndex setIndex, BindingIndex bindIndex,
                       const VulkanHostMemoryBuffer &memBuffer) {
    std::lock_guard<std::mutex> lock(mutex);
    vulkanRuntime.setResourceData(setIndex, bindIndex, memBuffer);
  }

  void setEntryPoint(const char *entryPoint) {
    std::lock_guard<std::mutex> lock(mutex);
    vulkanRuntime.setEntryPoint(entryPoint);
  }

  void setNumWorkGroups(NumWorkGroups numWorkGroups) {
    std::lock_guard<std::mutex> lock(mutex);
    vulkanRuntime.setNumWorkGroups(numWorkGroups);
  }

  void setShaderModule(uint8_t *shader, uint32_t size) {
    std::lock_guard<std::mutex> lock(mutex);
    vulkanRuntime.setShaderModule(shader, size);
  }

  void runOnVulkan() {
    std::lock_guard<std::mutex> lock(mutex);
    if (failed(vulkanRuntime.initRuntime()) || failed(vulkanRuntime.run()) ||
        failed(vulkanRuntime.updateHostMemoryBuffers()) ||
        failed(vulkanRuntime.destroy())) {
      llvm::errs() << "runOnVulkan failed";
    }
  }

private:
  VulkanRuntime vulkanRuntime;
  std::mutex mutex;
};

} // namespace

template <typename T, int N>
struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  int64_t offset;
  int64_t sizes[N];
  int64_t strides[N];
};

extern "C" {

VULKAN_RT_EXPORT void *initVulkan();
VULKAN_RT_EXPORT void deinitVulkan(void *vkRuntimeManager);
VULKAN_RT_EXPORT void runOnVulkan(void *vkRuntimeManager);
VULKAN_RT_EXPORT void setEntryPoint(void *vkRuntimeManager,
                                    const char *entryPoint);
VULKAN_RT_EXPORT void setNumWorkGroups(void *vkRuntimeManager, uint32_t x,
                                       uint32_t y, uint32_t z);
VULKAN_RT_EXPORT void setBinaryShader(void *vkRuntimeManager, uint8_t *shader,
                                      uint32_t size);
VULKAN_RT_EXPORT void bindMemRef1DFloat(void *vkRuntimeManager,
                                        DescriptorSetIndex setIndex,
                                        BindingIndex bindIndex,
                                        MemRefDescriptor<float, 1> *ptr);
VULKAN_RT_EXPORT void
_mlir_ciface_fillResource1DFloat(MemRefDescriptor<float, 1> *ptr, float value);

/// Initializes `VulkanRuntimeManager` and returns a pointer to it.
void *initVulkan() { return new VulkanRuntimeManager(); }

/// Deinitializes `VulkanRuntimeManager` by the given pointer.
void deinitVulkan(void *vkRuntimeManager) {
  delete reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager);
}

void runOnVulkan(void *vkRuntimeManager) {
  reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager)->runOnVulkan();
}

void setEntryPoint(void *vkRuntimeManager, const char *entryPoint) {
  reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager)
      ->setEntryPoint(entryPoint);
}

void setNumWorkGroups(void *vkRuntimeManager, uint32_t x, uint32_t y,
                      uint32_t z) {
  reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager)
      ->setNumWorkGroups({x, y, z});
}

void setBinaryShader(void *vkRuntimeManager, uint8_t *shader, uint32_t size) {
  reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager)
      ->setShaderModule(shader, size);
}

/// Binds the given 1D float memref to the given descriptor set and descriptor
/// index.
void bindMemRef1DFloat(void *vkRuntimeManager, DescriptorSetIndex setIndex,
                       BindingIndex bindIndex,
                       MemRefDescriptor<float, 1> *ptr) {
  VulkanHostMemoryBuffer memBuffer{
      ptr->allocated, static_cast<uint32_t>(ptr->sizes[0] * sizeof(float))};
  reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager)
      ->setResourceData(setIndex, bindIndex, memBuffer);
}

/// Fills the given 1D float memref with the given float value.
void _mlir_ciface_fillResource1DFloat(MemRefDescriptor<float, 1> *ptr,
                                      float value) {
  std::fill_n(ptr->allocated, ptr->sizes[0], value);
}

} // extern "C"
