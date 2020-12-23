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
  void *result =
      new LevelZeroInvocation(static_cast<LevelZeroDevice *>(device));
  IVLOG(2, "  ->" << result);
  instance = static_cast<LevelZeroInvocation *>(result);
  return result;
}

void level_zero_destroy_execenv(void *invocation) {
  IVLOG(2, "level_zero_destroy_execenv env=" << invocation);
  delete static_cast<LevelZeroInvocation *>(invocation);
  instance = nullptr;
}
/*
void *levelZeroAlloc(void *invocation, size_t bytes) {
  return static_cast<LevelZeroInvocation *>(invocation)->allocateMemory(bytes);
}

void levelZeroSubmit(void *invocation);
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
*/
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

/*
void levelZeroAddKernelDep(void *kernel, void *event) {
  static_cast<LevelZeroKernel *>(kernel)->addDependency(
      static_cast<LevelZeroEvent *>(event));
}

void levelZeroSetKernelArg(void *kernel, uint32_t idx, void *memory) {
  static_cast<LevelZeroKernel *>(kernel)->setArg(
      idx, static_cast<LevelZeroMemory *>(memory));
}
*/
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
/*
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
*/
void level_zero_submit(void *invocation) {
  IVLOG(2, "level_zero_submit env=" << invocation);
  static_cast<LevelZeroInvocation *>(invocation)->flush();
}

void *level_zero_alloc(void *invocation, size_t bytes) {
  IVLOG(2, "level_zero_alloc env=" << invocation << ", size=" << bytes);
  void *result =
      static_cast<LevelZeroInvocation *>(invocation)->allocateMemory(bytes);
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
  void *result = static_cast<LevelZeroInvocation *>(invocation)
                     ->enqueueRead(static_cast<LevelZeroMemory *>(dev), host,
                                   dependencies);
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
  void *result = static_cast<LevelZeroInvocation *>(invocation)
                     ->enqueueWrite(static_cast<LevelZeroMemory *>(dev), host,
                                    dependencies);
  IVLOG(2, "  ->" << result);
  return result;
}

void level_zero_wait(uint32_t count, ...) {
  // No submit created in IR, so call by hand
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
