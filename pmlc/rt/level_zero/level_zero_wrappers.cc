// Copyright 2020 Intel Corporation

#include <cstdarg>
#include <vector>

#include "pmlc/rt/level_zero/level_zero_invocation.h"
#include "pmlc/rt/symbol_registry.h"
#include "pmlc/util/logging.h"

namespace pmlc::rt::level_zero {

static LevelZeroInvocation *instance = nullptr;

extern "C" {

void *level_zero_create_execenv(void *device) {
  IVLOG(2, "level_zero_create_execenv device=" << device);
  ze_command_queue_group_properties_t p;
  p.flags = ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE;
  void *result =
      new LevelZeroInvocation(static_cast<LevelZeroDevice *>(device), p);
  IVLOG(2, "  ->" << result);
  instance = static_cast<LevelZeroInvocation *>(result);
  return result;
}

void level_zero_destroy_execenv(void *invocation) {
  IVLOG(2, "level_zero_destroy_execenv env=" << invocation);
  delete static_cast<LevelZeroInvocation *>(invocation);
  instance = nullptr;
}

void *level_zero_create_kernel(void *invocation, char *binary, uint32_t bytes,
                               const char *name) {
  IVLOG(2, "level_zero_create_kernel env=" << invocation << ", size=" << bytes
                                           << ", name=" << name);
  void *result = static_cast<LevelZeroInvocation *>(invocation)
                     ->createKernelFromIL(binary, bytes, name);
  IVLOG(2, "  ->" << result);
  return result;
}

void level_zero_destroy_kernel(void *invocation, void *kernel) {
  IVLOG(2, "level_zero_destroy_execenv env=" << invocation
                                             << ", kernel=" << kernel);
  delete static_cast<LevelZeroKernel *>(kernel);
}

void *level_zero_schedule_compute(void *invocation, void *kernel, uint64_t gws0,
                                  uint64_t gws1, uint64_t gws2, uint64_t lws0,
                                  uint64_t lws1, uint64_t lws2,
                                  uint64_t bufferCount, uint64_t eventCount,
                                  ...) {
  IVLOG(2, "level_zero_schedule_compute env=" << invocation
                                              << ", kernel=" << kernel);
  IVLOG(2, "  groupCount=(" << gws0 << ", " << gws1 << ", " << gws2 << ")");
  IVLOG(2, "  groupSize=(" << lws0 << ", " << lws1 << ", " << lws2 << ")");
  va_list args;
  va_start(args, eventCount);
  auto levelZeroKernel = static_cast<LevelZeroKernel *>(kernel);
  for (size_t i = 0; i < bufferCount; i++) {
    LevelZeroMemory *mem = va_arg(args, LevelZeroMemory *);
    IVLOG(2, "  arg[" << i << "]=" << mem);
    levelZeroKernel->setArg(i, mem);
  }
  for (size_t i = 0; i < eventCount; i++) {
    LevelZeroEvent *evt = va_arg(args, LevelZeroEvent *);
    IVLOG(2, "  event=" << evt);
    levelZeroKernel->addDependency(evt);
  }
  va_end(args);
  ze_group_count_t gws = {static_cast<uint32_t>(gws0),
                          static_cast<uint32_t>(gws1),
                          static_cast<uint32_t>(gws2)};
  ze_group_count_t lws = {static_cast<uint32_t>(lws0),
                          static_cast<uint32_t>(lws1),
                          static_cast<uint32_t>(lws2)};
  void *result =
      static_cast<LevelZeroInvocation *>(invocation)
          ->enqueueKernel(static_cast<LevelZeroKernel *>(kernel), gws, lws);
  IVLOG(2, "  ->" << result);
  return result;
}

void level_zero_submit(void *invocation) {
  IVLOG(2, "level_zero_submit env=" << invocation);
  static_cast<LevelZeroInvocation *>(invocation)->flush();
}

void *level_zero_alloc(void *invocation, size_t bytes) {
  IVLOG(2, "level_zero_alloc env=" << invocation << ", size=" << bytes);
  // Can add specific optimization to switch memory kind(Host, Device, Shared)
  // Event takes more time than computation on some machines.
  void *result = static_cast<LevelZeroInvocation *>(invocation)
                     ->allocateMemory(bytes, LevelZeroMemoryKind::Host);
  IVLOG(2, "  ->" << result);
  return result;
}

void level_zero_dealloc(void *invocation, void *memory) {
  IVLOG(2, "level_zero_dealloc env=" << invocation << ", buffer=" << memory);
  static_cast<LevelZeroInvocation *>(invocation)
      ->deallocateMemory(static_cast<LevelZeroMemory *>(memory));
}

void *level_zero_schedule_read(void *host, void *dev, void *invocation,
                               uint32_t count, ...) {
  std::vector<LevelZeroEvent *> dependencies;
  va_list args;
  va_start(args, count);
  for (unsigned i = 0; i < count; ++i) {
    LevelZeroEvent *evt = va_arg(args, LevelZeroEvent *);
    IVLOG(2, "  event=" << evt);
    dependencies.push_back(evt);
  }
  va_end(args);
  LevelZeroMemory *src = static_cast<LevelZeroMemory *>(dev);
  if (src->getKind() == LevelZeroMemoryKind::Host) {
    level_zero_submit(invocation);
  }
  void *result = static_cast<LevelZeroInvocation *>(invocation)
                     ->enqueueRead(src, host, dependencies);
  IVLOG(2, "  ->" << result);
  return result;
}

void *level_zero_schedule_write(void *host, void *dev, void *invocation,
                                uint32_t count, ...) {
  IVLOG(2, "level_zero_schedule_write env=" << invocation << ", host=" << host
                                            << ", dev=" << dev);
  std::vector<LevelZeroEvent *> dependencies;
  va_list args;
  va_start(args, count);
  for (unsigned i = 0; i < count; ++i) {
    LevelZeroEvent *evt = va_arg(args, LevelZeroEvent *);
    IVLOG(2, "  event=" << evt);
    dependencies.push_back(evt);
  }
  va_end(args);
  LevelZeroMemory *dst = static_cast<LevelZeroMemory *>(dev);
  if (dst->getKind() == LevelZeroMemoryKind::Host) {
    level_zero_submit(invocation);
  }
  void *result = static_cast<LevelZeroInvocation *>(invocation)
                     ->enqueueWrite(dst, host, dependencies);
  IVLOG(2, "  ->" << result);
  return result;
}

void level_zero_wait(uint32_t count, ...) {
  // No submit command created in IR, so call by hand
  level_zero_submit(instance);

  IVLOG(2, "level_zero_wait");
  std::vector<LevelZeroEvent *> events;
  va_list args;
  va_start(args, count);
  for (unsigned i = 0; i < count; ++i) {
    LevelZeroEvent *evt = va_arg(args, LevelZeroEvent *);
    IVLOG(2, "  event=" << evt);
    events.push_back(evt);
  }
  va_end(args);
  LevelZeroEvent::wait(events);
}

void *level_zero_schedule_barrier(void *invocation, uint32_t count, ...) {
  IVLOG(2, "level_zero_schedule_barrier env=" << invocation);
  std::vector<LevelZeroEvent *> dependencies;
  va_list args;
  va_start(args, count);
  for (unsigned i = 0; i < count; ++i) {
    LevelZeroEvent *evt = va_arg(args, LevelZeroEvent *);
    IVLOG(2, "  event=" << evt);
    dependencies.push_back(evt);
  }
  va_end(args);
  return static_cast<LevelZeroInvocation *>(invocation)
      ->enqueueBarrier(dependencies);
}

} // extern "C"

void registerSymbols() {
  using pmlc::rt::registerSymbol;

  // LevelZero Runtime functions
#define REG(x) registerSymbol(#x, reinterpret_cast<void *>(x));
  REG(level_zero_create_execenv)
  REG(level_zero_destroy_execenv)
  REG(level_zero_create_kernel)
  REG(level_zero_destroy_kernel)
  REG(level_zero_schedule_compute)
  REG(level_zero_submit)
  REG(level_zero_alloc)
  REG(level_zero_dealloc)
  REG(level_zero_schedule_write)
  REG(level_zero_schedule_read)
  REG(level_zero_wait)
  REG(level_zero_schedule_barrier)
#undef REG
}

} // namespace pmlc::rt::level_zero
