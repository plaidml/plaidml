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

#include "mlir/ExecutionEngine/RunnerUtils.h"

#include "pmlc/compiler/registry.h"
#include "pmlc/rt/vulkan/vulkan_runtime.h"

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
    vulkanRuntime.createLaunchKernelAction(shader, size, entryPoint,
                                           numWorkGroups);
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

extern "C" {
#define UNUSED(x) (void)(x)

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

void bindBuffer(void *vkRuntimeManager, DescriptorSetIndex setIndex,
                BindingIndex bindIndex, uint32_t bufferByteSize, int64_t rank,
                void **ptr) {
  UNUSED(rank);
  VulkanHostMemoryBuffer memBuffer{*ptr, bufferByteSize};
  reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager)
      ->setResourceData(setIndex, bindIndex, memBuffer);
}

void fillResourceFloat32(int64_t rank, void **ptr, int32_t size, float value) {
  UNUSED(rank);
  std::fill_n(reinterpret_cast<float *>(*ptr), size, value);
}
} // extern "C"

namespace {
struct Registration {
  Registration() {
    using pmlc::compiler::registerSymbol;

    // Vulkan Runtime functions
    registerSymbol("initVulkan", reinterpret_cast<void *>(initVulkan));
    registerSymbol("deinitVulkan", reinterpret_cast<void *>(deinitVulkan));
    registerSymbol("createVulkanLaunchKernelAction",
                   reinterpret_cast<void *>(createVulkanLaunchKernelAction));
    registerSymbol("createVulkanMemoryTransferAction",
                   reinterpret_cast<void *>(createVulkanMemoryTransferAction));
    registerSymbol("setVulkanLaunchKernelAction",
                   reinterpret_cast<void *>(setVulkanLaunchKernelAction));
    registerSymbol("addVulkanLaunchActionToSchedule",
                   reinterpret_cast<void *>(addVulkanLaunchActionToSchedule));
    registerSymbol("submitCommandBuffers",
                   reinterpret_cast<void *>(submitCommandBuffers));
    registerSymbol("bindBufferBFloat16", reinterpret_cast<void *>(bindBuffer));
    registerSymbol("bindBufferFloat16", reinterpret_cast<void *>(bindBuffer));
    registerSymbol("bindBufferFloat32", reinterpret_cast<void *>(bindBuffer));
    registerSymbol("bindBufferFloat64", reinterpret_cast<void *>(bindBuffer));
    registerSymbol("bindBufferInteger8", reinterpret_cast<void *>(bindBuffer));
    registerSymbol("bindBufferInteger16", reinterpret_cast<void *>(bindBuffer));
    registerSymbol("bindBufferInteger32", reinterpret_cast<void *>(bindBuffer));
    registerSymbol("bindBufferInteger64", reinterpret_cast<void *>(bindBuffer));
    registerSymbol("fillResourceFloat32",
                   reinterpret_cast<void *>(fillResourceFloat32));
  }
};
static Registration reg;
} // namespace
