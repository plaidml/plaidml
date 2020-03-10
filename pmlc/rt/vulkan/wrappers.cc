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

#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/raw_ostream.h"

#include "pmlc/rt/vulkan/vulkan_runtime.h"

namespace {

// TODO(denis0x0D): This static machinery should be replaced by `initVulkan` and
// `deinitVulkan` to be more explicit and to avoid static initialization and
// destruction.
class VulkanRuntimeManager;
static llvm::ManagedStatic<VulkanRuntimeManager> vkRuntimeManager;

class VulkanRuntimeManager {
public:
  VulkanRuntimeManager() {
    int64_t buffer_size = 9;
    value[0] = reinterpret_cast<float *>(malloc(buffer_size * sizeof(float)));
    value[1] = reinterpret_cast<float *>(malloc(buffer_size * sizeof(float)));
    value[2] = reinterpret_cast<float *>(malloc(buffer_size * sizeof(float)));
    for (int i = 0; i < 9; i++) {
      *(value[0] + i) = i + 1;
      *(value[1] + i) = i + 1;
      *(value[2] + i) = 0;
    }
  }
  VulkanRuntimeManager(const VulkanRuntimeManager &) = delete;
  VulkanRuntimeManager operator=(const VulkanRuntimeManager &) = delete;
  ~VulkanRuntimeManager() {
    free(value[0]);
    free(value[1]);
    free(value[2]);
  }

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
  float *value[3];

private:
  VulkanRuntime vulkanRuntime;
  std::mutex mutex;
};

} // namespace

extern "C" {
/// Fills the given memref with the given value.
/// Binds the given memref to the given descriptor set and descriptor index.
void setResourceData(const DescriptorSetIndex setIndex, BindingIndex bindIndex,
                     float *allocated, float *aligned, int64_t offset,
                     int64_t size, int64_t stride, float value) {
  std::fill_n(allocated, size, value);
  VulkanHostMemoryBuffer memBuffer{allocated,
                                   static_cast<uint32_t>(size * sizeof(float))};
  vkRuntimeManager->setResourceData(setIndex, bindIndex, memBuffer);
}

void setResourceData2D(const DescriptorSetIndex setIndex,
                       BindingIndex bindIndex, float *allocated, float *aligned,
                       int64_t offset, int64_t size_0, int64_t size_1,
                       int64_t stride_0, int64_t stride_1,
                       int64_t buffer_index) {
  int64_t size = size_0 * size_1;
  memcpy(allocated, vkRuntimeManager->value[buffer_index],
         size * sizeof(float));
  VulkanHostMemoryBuffer memBuffer{allocated,
                                   static_cast<uint32_t>(size * sizeof(float))};
  vkRuntimeManager->setResourceData(setIndex, bindIndex, memBuffer);
}

void setEntryPoint(const char *entryPoint) {
  vkRuntimeManager->setEntryPoint(entryPoint);
}

void setNumWorkGroups(uint32_t x, uint32_t y, uint32_t z) {
  vkRuntimeManager->setNumWorkGroups({x, y, z});
}

void setBinaryShader(uint8_t *shader, uint32_t size) {
  vkRuntimeManager->setShaderModule(shader, size);
}

void runOnVulkan() { vkRuntimeManager->runOnVulkan(); }
}
