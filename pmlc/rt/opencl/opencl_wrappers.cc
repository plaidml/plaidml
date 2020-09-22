// Copyright 2020 Intel Corporation

#include <cstdarg>
#include <vector>

#include "pmlc/rt/opencl/opencl_invocation.h"
#include "pmlc/rt/symbol_registry.h"
#include "pmlc/util/logging.h"

namespace pmlc::rt::opencl {

extern "C" {

void *oclCreate() { return new OpenCLInvocation(); }

void oclDestroy(void *invocation) {
  delete static_cast<OpenCLInvocation *>(invocation);
}

void *oclAlloc(void *invocation, uint32_t bytes, void *hostPtr) {
  return static_cast<OpenCLInvocation *>(invocation)
      ->allocateMemory(bytes, hostPtr);
}

void oclDealloc(void *invocation, void *memory) {
  static_cast<OpenCLInvocation *>(invocation)
      ->deallocateMemory(static_cast<OpenCLMemory *>(memory));
}

void *oclRead(void *dst, void *src, void *invocation, uint32_t count, ...) {
  std::vector<OpenCLEvent *> dependencies;
  va_list args;
  va_start(args, count);
  for (unsigned i = 0; i < count; ++i)
    dependencies.push_back(va_arg(args, OpenCLEvent *));
  va_end(args);
  return static_cast<OpenCLInvocation *>(invocation)
      ->enqueueRead(static_cast<OpenCLMemory *>(src), dst, dependencies);
}

void *oclWrite(void *src, void *dst, void *invocation, uint32_t count, ...) {
  std::vector<OpenCLEvent *> dependencies;
  va_list args;
  va_start(args, count);
  for (unsigned i = 0; i < count; ++i)
    dependencies.push_back(va_arg(args, OpenCLEvent *));
  va_end(args);
  return static_cast<OpenCLInvocation *>(invocation)
      ->enqueueWrite(static_cast<OpenCLMemory *>(dst), src, dependencies);
}

void *oclCreateKernel(void *invocation, char *binary, uint32_t bytes,
                      const char *name) {
  return static_cast<OpenCLInvocation *>(invocation)
      ->createKernelFromIL(binary, bytes, name);
}

void oclAddKernelDep(void *kernel, void *event) {
  static_cast<OpenCLKernel *>(kernel)->addDependency(
      static_cast<OpenCLEvent *>(event));
}

void oclSetKernelArg(void *kernel, uint32_t idx, void *memory) {
  static_cast<OpenCLKernel *>(kernel)->setArg(
      idx, static_cast<OpenCLMemory *>(memory));
}

void *oclScheduleFunc(void *invocation, void *kernel, uint64_t gws0,
                      uint64_t gws1, uint64_t gws2, uint64_t lws0,
                      uint64_t lws1, uint64_t lws2) {
  cl::NDRange gws(gws0, gws1, gws2);
  cl::NDRange lws(lws0, lws1, lws2);
  return static_cast<OpenCLInvocation *>(invocation)
      ->enqueueKernel(static_cast<OpenCLKernel *>(kernel), gws, lws);
}

void *oclBarrier(void *invocation, uint32_t count, ...) {
  std::vector<OpenCLEvent *> dependencies;
  va_list args;
  va_start(args, count);
  for (unsigned i = 0; i < count; ++i)
    dependencies.push_back(va_arg(args, OpenCLEvent *));
  va_end(args);
  return static_cast<OpenCLInvocation *>(invocation)
      ->enqueueBarrier(dependencies);
}

void oclSubmit(void *invocation) {
  static_cast<OpenCLInvocation *>(invocation)->flush();
}

void oclWait(uint32_t count, ...) {
  std::vector<OpenCLEvent *> events;
  va_list args;
  va_start(args, count);
  for (unsigned i = 0; i < count; ++i)
    events.push_back(va_arg(args, OpenCLEvent *));
  va_end(args);
  OpenCLEvent::wait(events);
}

} // extern "C"

namespace {
struct Registration {
  Registration() {
    using pmlc::rt::registerSymbol;

    // OpenCL Runtime functions
    registerSymbol("oclCreate", reinterpret_cast<void *>(oclCreate));
    registerSymbol("oclDestroy", reinterpret_cast<void *>(oclDestroy));
    registerSymbol("oclAlloc", reinterpret_cast<void *>(oclAlloc));
    registerSymbol("oclDealloc", reinterpret_cast<void *>(oclDealloc));
    registerSymbol("oclRead", reinterpret_cast<void *>(oclRead));
    registerSymbol("oclWrite", reinterpret_cast<void *>(oclWrite));
    registerSymbol("oclCreateKernel",
                   reinterpret_cast<void *>(oclCreateKernel));
    registerSymbol("oclSetKernelArg",
                   reinterpret_cast<void *>(oclSetKernelArg));
    registerSymbol("oclAddKernelDep",
                   reinterpret_cast<void *>(oclAddKernelDep));
    registerSymbol("_mlir_ciface_oclScheduleFunc",
                   reinterpret_cast<void *>(oclScheduleFunc));
    registerSymbol("oclBarrier", reinterpret_cast<void *>(oclBarrier));
    registerSymbol("oclSubmit", reinterpret_cast<void *>(oclSubmit));
    registerSymbol("oclWait", reinterpret_cast<void *>(oclWait));
  }
};
static Registration reg;
} // namespace
} // namespace pmlc::rt::opencl
