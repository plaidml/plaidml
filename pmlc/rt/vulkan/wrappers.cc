// Vulkan runtime wrappers, originally from the LLVM project, and subsequently
// modified by Intel Corporation.
//
// Original copyright:
//
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

#include "pmlc/rt/symbol_registry.h"
#include "pmlc/rt/vulkan/vulkan_invocation.h"

using pmlc::rt::Device;

namespace pmlc::rt::vulkan {

extern "C" {

void *vkInit(void *device) {
  return new VulkanInvocation(static_cast<VulkanDevice *>(device));
}

void vkDeinit(void *vkInvocation) {
  delete static_cast<VulkanInvocation *>(vkInvocation);
}

void vkCreateLaunchKernelAction(void *vkInvocation, uint8_t *shader,
                                uint32_t size, const char *entryPoint,
                                uint32_t x, uint32_t y, uint32_t z) {
  static_cast<VulkanInvocation *>(vkInvocation)
      ->createLaunchKernelAction(shader, size, entryPoint, {x, y, z});
}

void vkCreateMemoryTransferAction(void *vkInvocation, uint64_t src_index,
                                  uint64_t src_binding, uint64_t dst_index,
                                  uint64_t dst_binding) {
  static_cast<VulkanInvocation *>(vkInvocation)
      ->createMemoryTransferAction(src_index, src_binding, dst_index,
                                   dst_binding);
}

void vkSetLaunchKernelAction(void *vkInvocation, uint32_t subgroupSize) {
  static_cast<VulkanInvocation *>(vkInvocation)
      ->setLaunchKernelAction(subgroupSize);
}

void vkRun(void *vkInvocation) {
  static_cast<VulkanInvocation *>(vkInvocation)->run();
}

void *vkScheduleFunc(void *vkInvocation) {
  static_cast<VulkanInvocation *>(vkInvocation)->addLaunchActionToSchedule();
  return nullptr;
}

void vkWait(uint32_t count, ...) {
  // TODO(Yanglei Zou): replace vkCmdPipelineBarrier by
  // vkCmdSetEvent + vkCmdWaitEvents
}

void vkBindBuffer(void *vkInvocation, DescriptorSetIndex setIndex,
                  BindingIndex bindIndex, size_t bufferByteSize, void *ptr) {
  VulkanHostMemoryBuffer memBuffer{ptr, static_cast<uint32_t>(bufferByteSize)};
  static_cast<VulkanInvocation *>(vkInvocation)
      ->setResourceData(setIndex, bindIndex, memBuffer);
}

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
    using pmlc::rt::registerSymbol;

    // Vulkan Runtime functions
    registerSymbol("vkInit", reinterpret_cast<void *>(vkInit));
    registerSymbol("vkDeinit", reinterpret_cast<void *>(vkDeinit));
    registerSymbol("vkCreateLaunchKernelAction",
                   reinterpret_cast<void *>(vkCreateLaunchKernelAction));
    registerSymbol("vkCreateMemoryTransferAction",
                   reinterpret_cast<void *>(vkCreateMemoryTransferAction));
    registerSymbol("vkSetLaunchKernelAction",
                   reinterpret_cast<void *>(vkSetLaunchKernelAction));
    registerSymbol("vkRun", reinterpret_cast<void *>(vkRun));
    registerSymbol("vkWait", reinterpret_cast<void *>(vkWait));
    registerSymbol("vkScheduleFunc", reinterpret_cast<void *>(vkScheduleFunc));
    registerSymbol("vkBindBuffer", reinterpret_cast<void *>(vkBindBuffer));
    registerSymbol("_mlir_ciface_fillResourceFloat32",
                   reinterpret_cast<void *>(_mlir_ciface_fillResourceFloat32));
  }
};
static Registration reg;
} // namespace
} // namespace pmlc::rt::vulkan
