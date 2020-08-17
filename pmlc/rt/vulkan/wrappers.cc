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

#include "half.hpp"
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
    vulkanRuntime.createLaunchKernelAction(shader, size, entryPoint,
                                           numWorkGroups);
  }

  void createMemoryTransferAction(uint64_t src_index, uint64_t src_binding,
                                  uint64_t dst_index, uint64_t dst_binding) {
    vulkanRuntime.createMemoryTransferAction(src_index, src_binding, dst_index,
                                             dst_binding);
  }

  void setResourceData(DescriptorSetIndex setIndex, BindingIndex bindIndex,
                       const VulkanHostMemoryBuffer &memBuffer) {
    vulkanRuntime.setResourceData(setIndex, bindIndex, memBuffer);
  }

  void setLaunchKernelAction() {
    if (failed(vulkanRuntime.setLaunchKernelAction())) {
      llvm::errs() << "runOnVulkan failed";
    }
  }

  void addLaunchActionToSchedule() {
    vulkanRuntime.addLaunchActionToSchedule();
  }

  void submitCommandBuffers() {
    if (failed(vulkanRuntime.submitCommandBuffers())) {
      llvm::errs() << "vulkanRuntime.submitBuffer() failed";
    }
  }

private:
  VulkanRuntime vulkanRuntime;
};

template <typename T>
void bindBuffer(void *vkRuntimeManager, DescriptorSetIndex setIndex,
                BindingIndex bindIndex, uint32_t bufferByteSize,
                ::UnrankedMemRefType<T> *unrankedMemRef) {
  DynamicMemRefType<T> memRef(*unrankedMemRef);
  T *ptr = memRef.data + memRef.offset;
  VulkanHostMemoryBuffer memBuffer{ptr, bufferByteSize};
  reinterpret_cast<VulkanRuntimeManager *>(vkRuntimeManager)
      ->setResourceData(setIndex, bindIndex, memBuffer);
}

} // namespace

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

#define BIND_BUFFER_IMPL(_name_, _type_)                                       \
  void _mlir_ciface_bindBuffer##_name_(                                        \
      void *vkRuntimeManager, DescriptorSetIndex setIndex,                     \
      BindingIndex bindIndex, uint32_t bufferByteSize,                         \
      ::UnrankedMemRefType<_type_> *unrankedMemRef) {                          \
    bindBuffer(vkRuntimeManager, setIndex, bindIndex, bufferByteSize,          \
               unrankedMemRef);                                                \
  }

BIND_BUFFER_IMPL(Float16, half_float::half);
BIND_BUFFER_IMPL(Float32, float);
BIND_BUFFER_IMPL(Float64, double);
BIND_BUFFER_IMPL(Integer8, int8_t);
BIND_BUFFER_IMPL(Integer16, int16_t);
BIND_BUFFER_IMPL(Integer32, int32_t);
BIND_BUFFER_IMPL(Integer64, int64_t);

void _mlir_ciface_fillResourceFloat32(
    ::UnrankedMemRefType<float> *unrankedMemRef, int32_t size, float value) {
  DynamicMemRefType<float> memRef(*unrankedMemRef);
  float *ptr = memRef.data + memRef.offset;
  std::fill_n(ptr, size, value);
}

} // extern "C"

namespace {
struct Registration {
  Registration() {
    using pmlc::compiler::registerSymbol;

    // Vulkan Runtime functions
    registerSymbol("initVulkan", reinterpret_cast<void *>(initVulkan));
    registerSymbol("deinitVulkan", reinterpret_cast<void *>(deinitVulkan));
    registerSymbol("_mlir_ciface_createVulkanLaunchKernelAction",
                   reinterpret_cast<void *>(createVulkanLaunchKernelAction));
    registerSymbol("createVulkanMemoryTransferAction",
                   reinterpret_cast<void *>(createVulkanMemoryTransferAction));
    registerSymbol("setVulkanLaunchKernelAction",
                   reinterpret_cast<void *>(setVulkanLaunchKernelAction));
    registerSymbol("addVulkanLaunchActionToSchedule",
                   reinterpret_cast<void *>(addVulkanLaunchActionToSchedule));
    registerSymbol("submitCommandBuffers",
                   reinterpret_cast<void *>(submitCommandBuffers));
    registerSymbol("_mlir_ciface_bindBufferFloat16",
                   reinterpret_cast<void *>(_mlir_ciface_bindBufferFloat16));
    registerSymbol("_mlir_ciface_bindBufferFloat32",
                   reinterpret_cast<void *>(_mlir_ciface_bindBufferFloat32));
    registerSymbol("_mlir_ciface_bindBufferFloat64",
                   reinterpret_cast<void *>(_mlir_ciface_bindBufferFloat64));
    registerSymbol("_mlir_ciface_bindBufferInteger8",
                   reinterpret_cast<void *>(_mlir_ciface_bindBufferInteger8));
    registerSymbol("_mlir_ciface_bindBufferInteger16",
                   reinterpret_cast<void *>(_mlir_ciface_bindBufferInteger16));
    registerSymbol("_mlir_ciface_bindBufferInteger32",
                   reinterpret_cast<void *>(_mlir_ciface_bindBufferInteger32));
    registerSymbol("_mlir_ciface_bindBufferInteger64",
                   reinterpret_cast<void *>(_mlir_ciface_bindBufferInteger64));
    registerSymbol("_mlir_ciface_fillResourceFloat32",
                   reinterpret_cast<void *>(_mlir_ciface_fillResourceFloat32));
  }
};
static Registration reg;
} // namespace
