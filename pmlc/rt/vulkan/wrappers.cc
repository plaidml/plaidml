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

#include <cstdarg>
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
                                uint32_t x, uint32_t y, uint32_t z,
                                uint32_t count, ...) {
  std::vector<void *> deviceBuffers;
  va_list args;
  va_start(args, count);
  for (unsigned i = 0; i < count; ++i)
    deviceBuffers.push_back(va_arg(args, void *));
  va_end(args);

  static_cast<VulkanInvocation *>(vkInvocation)
      ->createLaunchKernelAction(shader, size, entryPoint, {x, y, z},
                                 deviceBuffers);
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

void *vkAlloc(void *vkInvocation, uint32_t bytes, void *hostPtr) {
  vulkanBuffer *newBuffer = new vulkanBuffer();
  VulkanHostMemoryBuffer memBuffer{hostPtr, bytes};
  newBuffer->HostBuffer = memBuffer;
  newBuffer->spirvClass = mlir::spirv::StorageClass::StorageBuffer;
  DescriptorSetIndex setIndex = 0;
  return static_cast<VulkanInvocation *>(vkInvocation)
      ->createMemoryBuffer(setIndex, newBuffer);
}

void vkDealloc(void *invocation, void *memory) {
  static_cast<VulkanInvocation *>(invocation)->deallocDeviceBuffer(memory);
}

void *vkRead(void *dst, void *src, void *invocation, uint32_t count, ...) {
  static_cast<VulkanInvocation *>(invocation)->copyDeviceBufferToHost(dst, src);
  // TODO: return Vulkan Event
  return nullptr;
}

void *vkWrite(void *src, void *dst, void *invocation, uint32_t count, ...) {
  static_cast<VulkanInvocation *>(invocation)->copyHostBufferToDevice(src, dst);
  // TODO: return Vulkan Event
  return nullptr;
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
    registerSymbol("vkSetLaunchKernelAction",
                   reinterpret_cast<void *>(vkSetLaunchKernelAction));
    registerSymbol("vkRun", reinterpret_cast<void *>(vkRun));
    registerSymbol("vkWait", reinterpret_cast<void *>(vkWait));
    registerSymbol("vkAlloc", reinterpret_cast<void *>(vkAlloc));
    registerSymbol("vkDealloc", reinterpret_cast<void *>(vkDealloc));
    registerSymbol("vkRead", reinterpret_cast<void *>(vkRead));
    registerSymbol("vkWrite", reinterpret_cast<void *>(vkWrite));
    registerSymbol("vkScheduleFunc", reinterpret_cast<void *>(vkScheduleFunc));
  }
};
static Registration reg;
} // namespace
} // namespace pmlc::rt::vulkan
