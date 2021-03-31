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

void *vkScheduleFunc(void *vkInvocation, uint32_t subgroupSize, uint8_t *shader,
                     uint32_t size, const char *entryPoint, uint32_t x,
                     uint32_t y, uint32_t z, uint32_t count, ...) {
  std::vector<void *> deviceBuffers;
  va_list args;
  va_start(args, count);
  for (unsigned i = 0; i < count; ++i)
    deviceBuffers.push_back(va_arg(args, void *));
  va_end(args);
  return static_cast<VulkanInvocation *>(vkInvocation)
      ->scheduleLaunchKernelAction(subgroupSize, shader, size, entryPoint,
                                   {x, y, z}, deviceBuffers);
}

void vkRun(void *vkInvocation) {
  static_cast<VulkanInvocation *>(vkInvocation)->run();
}

void vkWait(uint32_t count, ...) {
  std::vector<vulkanEvent *> events;
  va_list args;
  va_start(args, count);
  for (unsigned i = 0; i < count; ++i)
    events.push_back(va_arg(args, vulkanEvent *));
  va_end(args);

  if (events.size() > 0) {
    events[0]->invocation->createWaitEventsAction(events);
  }
}

void *vkAlloc(void *vkInvocation, uint32_t bytes, void *hostPtr) {
  return static_cast<VulkanInvocation *>(vkInvocation)
      ->createMemoryBuffer(bytes, hostPtr);
}

void vkDealloc(void *invocation, void *memory) {
  static_cast<VulkanInvocation *>(invocation)->deallocDeviceBuffer(memory);
}

void *vkRead(void *dst, void *src, void *invocation, uint32_t count, ...) {
  return static_cast<VulkanInvocation *>(invocation)
      ->copyDeviceBufferToHost(dst, src);
}

void *vkWrite(void *src, void *dst, void *invocation, uint32_t count, ...) {
  return static_cast<VulkanInvocation *>(invocation)
      ->copyHostBufferToDevice(src, dst);
}

} // extern "C"

void registerSymbols() {
  using pmlc::rt::registerSymbol;

  // Vulkan Runtime functions
  registerSymbol("vkInit", reinterpret_cast<void *>(vkInit));
  registerSymbol("vkDeinit", reinterpret_cast<void *>(vkDeinit));
  registerSymbol("vkRun", reinterpret_cast<void *>(vkRun));
  registerSymbol("vkWait", reinterpret_cast<void *>(vkWait));
  registerSymbol("vkAlloc", reinterpret_cast<void *>(vkAlloc));
  registerSymbol("vkDealloc", reinterpret_cast<void *>(vkDealloc));
  registerSymbol("vkRead", reinterpret_cast<void *>(vkRead));
  registerSymbol("vkWrite", reinterpret_cast<void *>(vkWrite));
  registerSymbol("vkScheduleFunc", reinterpret_cast<void *>(vkScheduleFunc));
}

} // namespace pmlc::rt::vulkan
