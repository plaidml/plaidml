// Copyright 2020 Intel Corporation

#include <cstdarg>
#include <vector>

#include "pmlc/rt/opencl/opencl_invocation.h"
#include "pmlc/rt/symbol_registry.h"
#include "pmlc/util/logging.h"

namespace pmlc::rt::opencl {

extern "C" {

void *ocl_create_execenv(void *device) {
  IVLOG(2, "ocl_create_execenv device=" << device);
  void *result = new OpenCLInvocation(static_cast<OpenCLDevice *>(device));
  IVLOG(2, "  ->" << result);
  return result;
}

void ocl_destroy_execenv(void *invocation) {
  IVLOG(2, "ocl_destroy_execenv env=" << invocation);
  delete static_cast<OpenCLInvocation *>(invocation);
}

void *ocl_create_kernel(void *invocation, char *binary, uint64_t bytes,
                        const char *name) {
  IVLOG(2, "ocl_create_kernel env=" << invocation << ", size=" << bytes
                                    << ", name=" << name);
  void *result = static_cast<OpenCLInvocation *>(invocation)
                     ->createKernelFromIL(binary, bytes, name);
  IVLOG(2, "  ->" << result);
  return result;
}

void ocl_destroy_kernel(void *invocation, void *kernel) {
  IVLOG(2, "ocl_destroy_execenv env=" << invocation << ", kernel=" << kernel);
  delete static_cast<OpenCLKernel *>(kernel);
}

void *ocl_schedule_compute(void *invocation, void *kernel,              //
                           uint64_t gws0, uint64_t gws1, uint64_t gws2, //
                           uint64_t lws0, uint64_t lws1, uint64_t lws2, //
                           uint64_t bufferCount, uint64_t eventCount, ...) {
  IVLOG(2, "ocl_schedule_compute env=" << invocation << ", kernel=" << kernel);
  IVLOG(2, "  grid=(" << gws0 << ", " << gws1 << ", " << gws2 << ")");
  IVLOG(2, "  block=(" << lws0 << ", " << lws1 << ", " << lws2 << ")");
  va_list args;
  va_start(args, eventCount);
  auto oclKernel = static_cast<OpenCLKernel *>(kernel);
  for (size_t i = 0; i < bufferCount; i++) {
    OpenCLMemory *mem = va_arg(args, OpenCLMemory *);
    IVLOG(2, "  arg[" << i << "]=" << mem);
    oclKernel->setArg(i, mem);
  }
  for (size_t i = 0; i < eventCount; i++) {
    OpenCLEvent *evt = va_arg(args, OpenCLEvent *);
    IVLOG(2, "  event=" << evt);
    oclKernel->addDependency(evt);
  }
  va_end(args);
  cl::NDRange gws(gws0 * lws0, gws1 * lws1, gws2 * lws2);
  cl::NDRange lws(lws0, lws1, lws2);
  void *result = static_cast<OpenCLInvocation *>(invocation)
                     ->enqueueKernel(oclKernel, gws, lws);
  IVLOG(2, "  ->" << result);
  return result;
}

void ocl_submit(void *invocation) {
  IVLOG(2, "ocl_submit env=" << invocation);
  static_cast<OpenCLInvocation *>(invocation)->flush();
}

void *ocl_alloc(void *invocation, size_t bytes) {
  IVLOG(2, "ocl_alloc env=" << invocation << ", size=" << bytes);
  void *result =
      static_cast<OpenCLInvocation *>(invocation)->allocateMemory(bytes);
  IVLOG(2, "  ->" << result);
  return result;
}

void ocl_dealloc(void *invocation, void *memory) {
  IVLOG(2, "ocl_dealloc env=" << invocation << ", buffer=" << memory);
  static_cast<OpenCLInvocation *>(invocation)
      ->deallocateMemory(static_cast<OpenCLMemory *>(memory));
}

void *ocl_schedule_read(void *host, void *dev, void *invocation, uint64_t count,
                        ...) {
  IVLOG(2, "ocl_schedule_read env=" << invocation << ", host=" << host
                                    << ", dev=" << dev);
  std::vector<OpenCLEvent *> dependencies;
  va_list args;
  va_start(args, count);
  for (unsigned i = 0; i < count; ++i) {
    OpenCLEvent *evt = va_arg(args, OpenCLEvent *);
    IVLOG(2, "  event=" << evt);
    dependencies.push_back(evt);
  }
  va_end(args);
  void *result =
      static_cast<OpenCLInvocation *>(invocation)
          ->enqueueRead(static_cast<OpenCLMemory *>(dev), host, dependencies);
  IVLOG(2, "  ->" << result);
  return result;
}

void *ocl_schedule_write(void *host, void *dev, void *invocation,
                         uint64_t count, ...) {
  IVLOG(2, "ocl_schedule_write env=" << invocation << ", host=" << host
                                     << ", dev=" << dev);
  std::vector<OpenCLEvent *> dependencies;
  va_list args;
  va_start(args, count);
  for (unsigned i = 0; i < count; ++i) {
    OpenCLEvent *evt = va_arg(args, OpenCLEvent *);
    IVLOG(2, "  event=" << evt);
    dependencies.push_back(evt);
  }
  va_end(args);
  void *result =
      static_cast<OpenCLInvocation *>(invocation)
          ->enqueueWrite(static_cast<OpenCLMemory *>(dev), host, dependencies);
  IVLOG(2, "  ->" << result);
  return result;
}

void ocl_wait(uint64_t count, ...) {
  IVLOG(2, "ocl_wait");
  std::vector<OpenCLEvent *> events;
  va_list args;
  va_start(args, count);
  for (unsigned i = 0; i < count; ++i) {
    OpenCLEvent *evt = va_arg(args, OpenCLEvent *);
    IVLOG(2, "  event=" << evt);
    events.push_back(evt);
  }
  va_end(args);
  OpenCLEvent::wait(events);
}

void *ocl_schedule_barrier(void *invocation, uint64_t count, ...) {
  IVLOG(2, "ocl_schedule_barrier env=" << invocation);
  std::vector<OpenCLEvent *> dependencies;
  va_list args;
  va_start(args, count);
  for (unsigned i = 0; i < count; ++i) {
    OpenCLEvent *evt = va_arg(args, OpenCLEvent *);
    IVLOG(2, "  event=" << evt);
    dependencies.push_back(evt);
  }
  va_end(args);
  return static_cast<OpenCLInvocation *>(invocation)
      ->enqueueBarrier(dependencies);
}

void ocl_dump_profiling(void *invocation) {
  IVLOG(2, "ocl_dump_profiling env = " << invocation);
  return static_cast<OpenCLInvocation *>(invocation)->finish();
}

} // extern "C"

void registerSymbols() {
  using pmlc::rt::registerSymbol;

  // OpenCL Runtime functions
#define REG(x) registerSymbol(#x, reinterpret_cast<void *>(x));
  REG(ocl_create_execenv)
  REG(ocl_destroy_execenv)
  REG(ocl_create_kernel)
  REG(ocl_destroy_kernel)
  REG(ocl_schedule_compute)
  REG(ocl_submit)
  REG(ocl_alloc)
  REG(ocl_dealloc)
  REG(ocl_schedule_write)
  REG(ocl_schedule_read)
  REG(ocl_wait)
  REG(ocl_schedule_barrier)
  REG(ocl_dump_profiling)
#undef REG
}

} // namespace pmlc::rt::opencl
