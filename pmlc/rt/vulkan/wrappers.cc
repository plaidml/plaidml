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
  VulkanRuntimeManager() {
    if (failed(vulkanRuntime.init())) {
      llvm::errs() << "vulkanRuntime.init() failed";
    }
  }
  VulkanRuntimeManager(const VulkanRuntimeManager &) = delete;
  VulkanRuntimeManager operator=(const VulkanRuntimeManager &) = delete;
  ~VulkanRuntimeManager() {
    if (failed(vulkanRuntime.destroy())) {
      llvm::errs() << "vulkanRuntime.destroy() failed";
    }
  }

  void createLaunchKernelAction(uint8_t *shader, uint32_t size,
                                const char *entryPoint,
                                NumWorkGroups numWorkGroups) {
    std::lock_guard<std::mutex> lock(mutex);
    vulkanRuntime.createLaunchKernelAction();
    vulkanRuntime.setShaderModule(shader, size);
    vulkanRuntime.setEntryPoint(entryPoint);
    vulkanRuntime.setNumWorkGroups(numWorkGroups);
  }

  void createMemoryTransferAction(uint64_t src_index, uint64_t src_binding,
                                  uint64_t dst_index, uint64_t dst_binding) {
    std::lock_guard<std::mutex> lock(mutex);
    vulkanRuntime.createMemoryTransferAction(src_index, src_binding, dst_index,
                                             dst_binding);
  }

  void setResourceData(DescriptorSetIndex setIndex, BindingIndex bindIndex,
                       const VulkanHostMemoryBuffer &memBuffer) {
    std::lock_guard<std::mutex> lock(mutex);
    vulkanRuntime.setResourceData(setIndex, bindIndex, memBuffer);
  }

  void setLaunchKernelAction() {
    std::lock_guard<std::mutex> lock(mutex);
    if (failed(vulkanRuntime.setLaunchKernelAction())) {
      llvm::errs() << "runOnVulkan failed";
    }
  }

  void addLaunchActionToSchedule() {
    std::lock_guard<std::mutex> lock(mutex);
    vulkanRuntime.addLaunchActionToSchedule();
  }

  void submitCommandBuffers() {
    std::lock_guard<std::mutex> lock(mutex);
    if (failed(vulkanRuntime.submitCommandBuffers())) {
      llvm::errs() << "vulkanRuntime.submitBuffer() failed";
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

template <typename T, int N>
void bindBuffer(void *vkRuntimeManager, DescriptorSetIndex setIndex,
                BindingIndex bindIndex, void *ptr) {
  auto descriptor = reinterpret_cast<MemRefDescriptor<T, N> *>(ptr);
  int64_t size = 1;
  for (int i = 0; i < N; i++) {
    size *= descriptor->sizes[i];
  }
  VulkanHostMemoryBuffer memBuffer{descriptor->allocated,
                                   static_cast<uint32_t>(size * sizeof(T))};
  reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager)
      ->setResourceData(setIndex, bindIndex, memBuffer);
}

extern "C" {
void *initVulkan() { return new VulkanRuntimeManager(); }

void deinitVulkan(void *vkRuntimeManager) {
  delete reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager);
}

void createVulkanLaunchKernelAction(void *vkRuntimeManager, uint8_t *shader,
                                    uint32_t size, const char *entryPoint,
                                    uint32_t x, uint32_t y, uint32_t z) {
  reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager)
      ->createLaunchKernelAction(shader, size, entryPoint, {x, y, z});
}

void createVulkanMemoryTransferAction(void *vkRuntimeManager,
                                      uint64_t src_index, uint64_t src_binding,
                                      uint64_t dst_index,
                                      uint64_t dst_binding) {
  reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager)
      ->createMemoryTransferAction(src_index, src_binding, dst_index,
                                   dst_binding);
}

void setVulkanLaunchKernelAction(void *vkRuntimeManager) {
  reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager)
      ->setLaunchKernelAction();
}

void addVulkanLaunchActionToSchedule(void *vkRuntimeManager) {
  reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager)
      ->addLaunchActionToSchedule();
}

void submitCommandBuffers(void *vkRuntimeManager) {
  reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager)
      ->submitCommandBuffers();
}

void bindBufferFloat32(void *vkRuntimeManager, DescriptorSetIndex setIndex,
                       BindingIndex bindIndex, int64_t rank, void *ptr) {
  switch (rank) {
  case 1:
    bindBuffer<float, 1>(vkRuntimeManager, setIndex, bindIndex, ptr);
    break;
  case 2:
    bindBuffer<float, 2>(vkRuntimeManager, setIndex, bindIndex, ptr);
    break;
  case 3:
    bindBuffer<float, 3>(vkRuntimeManager, setIndex, bindIndex, ptr);
    break;
  case 4:
    bindBuffer<float, 4>(vkRuntimeManager, setIndex, bindIndex, ptr);
    break;
  case 5:
    bindBuffer<float, 5>(vkRuntimeManager, setIndex, bindIndex, ptr);
    break;
  case 6:
    bindBuffer<float, 6>(vkRuntimeManager, setIndex, bindIndex, ptr);
    break;
  default:
    return;
  }
}

void bindBufferInt64(void *vkRuntimeManager, DescriptorSetIndex setIndex,
                     BindingIndex bindIndex, int64_t rank, void *ptr) {
  switch (rank) {
  case 1:
    bindBuffer<int64_t, 1>(vkRuntimeManager, setIndex, bindIndex, ptr);
    break;
  case 2:
    bindBuffer<int64_t, 2>(vkRuntimeManager, setIndex, bindIndex, ptr);
    break;
  case 3:
    bindBuffer<int64_t, 3>(vkRuntimeManager, setIndex, bindIndex, ptr);
    break;
  case 4:
    bindBuffer<int64_t, 4>(vkRuntimeManager, setIndex, bindIndex, ptr);
    break;
  case 5:
    bindBuffer<int64_t, 5>(vkRuntimeManager, setIndex, bindIndex, ptr);
    break;
  case 6:
    bindBuffer<int64_t, 6>(vkRuntimeManager, setIndex, bindIndex, ptr);
    break;
  default:
    return;
  }
}
} // extern "C"
