// Copyright 2020 Intel Corporation

#include <cstdarg>
#include <vector>

#include "pmlc/rt/level_zero/level_zero_invocation.h"
#include "pmlc/rt/symbol_registry.h"
#include "pmlc/util/logging.h"

namespace pmlc::rt::level_zero {

extern "C" {

void *levelZeroCreate(void *device) {
  return new LevelZeroInvocation(static_cast<LevelZeroDevice *>(device));
}

void levelZeroDestroy(void *invocation) {
  delete static_cast<LevelZeroInvocation *>(invocation);
}

void *levelZeroAlloc(void *invocation, size_t bytes) {
  return static_cast<LevelZeroInvocation *>(invocation)->allocateMemory(bytes);
}

void levelZeroDealloc(void *invocation, void *memory) {
  static_cast<LevelZeroInvocation *>(invocation)
      ->deallocateMemory(static_cast<LevelZeroMemory *>(memory));
}

void *levelZeroRead(void *dst, void *src, void *invocation, uint32_t count,
                    ...) {
  std::vector<LevelZeroEvent *> dependencies;
  va_list args;
  va_start(args, count);
  for (unsigned i = 0; i < count; ++i)
    dependencies.push_back(va_arg(args, LevelZeroEvent *));
  va_end(args);
  return static_cast<LevelZeroInvocation *>(invocation)
      ->enqueueRead(static_cast<LevelZeroMemory *>(src), dst, dependencies);
}

void *levelZeroWrite(void *src, void *dst, void *invocation, uint32_t count,
                     ...) {
  std::vector<LevelZeroEvent *> dependencies;
  va_list args;
  va_start(args, count);
  for (unsigned i = 0; i < count; ++i)
    dependencies.push_back(va_arg(args, LevelZeroEvent *));
  va_end(args);
  return static_cast<LevelZeroInvocation *>(invocation)
      ->enqueueWrite(static_cast<LevelZeroMemory *>(dst), src, dependencies);
}

void *levelZeroCreateKernel(void *invocation, char *binary, uint32_t bytes,
                            const char *name) {
  return static_cast<LevelZeroInvocation *>(invocation)
      ->createKernelFromIL(binary, bytes, name);
}

void levelZeroAddKernelDep(void *kernel, void *event) {
  static_cast<LevelZeroKernel *>(kernel)->addDependency(
      static_cast<LevelZeroEvent *>(event));
}

void levelZeroSetKernelArg(void *kernel, uint32_t idx, void *memory) {
  static_cast<LevelZeroKernel *>(kernel)->setArg(
      idx, static_cast<LevelZeroMemory *>(memory));
}

void *levelZeroScheduleFunc(void *invocation, void *kernel, uint64_t gws0,
                            uint64_t gws1, uint64_t gws2, uint64_t lws0,
                            uint64_t lws1, uint64_t lws2) {
  //ze_group_count_t gws(gws0, gws1, gws2);
  ze_group_count_t gws = {gws0 * lws0, gws1 * lws1, gws2 * lws2};
  ze_group_count_t lws = {lws0, lws1, lws2};
  return static_cast<LevelZeroInvocation *>(invocation)
      ->enqueueKernel(static_cast<LevelZeroKernel *>(kernel), gws, lws);
}

void *levelZeroBarrier(void *invocation, uint32_t count, ...) {
  std::vector<LevelZeroEvent *> dependencies;
  va_list args;
  va_start(args, count);
  for (unsigned i = 0; i < count; ++i)
    dependencies.push_back(va_arg(args, LevelZeroEvent *));
  va_end(args);
  return static_cast<LevelZeroInvocation *>(invocation)
      ->enqueueBarrier(dependencies);
}

void levelZeroSubmit(void *invocation) {
  static_cast<LevelZeroInvocation *>(invocation)->flush();
}

void levelZeroWait(uint32_t count, ...) {
  //TODO move to kernellaunch
  /*std::vector<LevelZeroEvent *> events;
  va_list args;
  va_start(args, count);
  for (unsigned i = 0; i < count; ++i)
    events.push_back(va_arg(args, LevelZeroEvent *));
  va_end(args);
  LevelZeroEvent::wait(events);*/
}

} // extern "C"

void registerSymbols() {
    using pmlc::rt::registerSymbol;

    // LevelZero Runtime functions
    registerSymbol("levelZeroCreate",
                   reinterpret_cast<void *>(levelZeroCreate));
    registerSymbol("levelZeroDestroy",
                   reinterpret_cast<void *>(levelZeroDestroy));
    registerSymbol("levelZeroAlloc", reinterpret_cast<void *>(levelZeroAlloc));
    registerSymbol("levelZeroDealloc",
                   reinterpret_cast<void *>(levelZeroDealloc));
    registerSymbol("levelZeroRead", reinterpret_cast<void *>(levelZeroRead));
    registerSymbol("levelZeroWrite", reinterpret_cast<void *>(levelZeroWrite));
    registerSymbol("levelZeroCreateKernel",
                   reinterpret_cast<void *>(levelZeroCreateKernel));
    registerSymbol("levelZeroSetKernelArg",
                   reinterpret_cast<void *>(levelZeroSetKernelArg));
    registerSymbol("levelZeroAddKernelDep",
                   reinterpret_cast<void *>(levelZeroAddKernelDep));
    registerSymbol("_mlir_ciface_levelZeroScheduleFunc",
                   reinterpret_cast<void *>(levelZeroScheduleFunc));
    registerSymbol("levelZeroBarrier",
                   reinterpret_cast<void *>(levelZeroBarrier));
    registerSymbol("levelZeroSubmit",
                   reinterpret_cast<void *>(levelZeroSubmit));
    registerSymbol("levelZeroWait", reinterpret_cast<void *>(levelZeroWait));
}

} // namespace pmlc::rt::level_zero
