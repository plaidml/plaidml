// Copyright 2020 Intel Corporation

#include <cstdarg>

#include "pmlc/rt/opencl/opencl_runtime.h"

#include "mlir/ExecutionEngine/RunnerUtils.h"
#include "llvm/Support/raw_ostream.h"

#include "pmlc/compiler/registry.h"

using namespace oclrt; // NOLINT

template <typename T>
T *wrapFailure(mlir::FailureOr<T *> result, std::string message) {
  if (mlir::failed(result)) {
    std::cout << message << " failed" << std::endl;
    return nullptr;
  }
  return result.getValue();
}

void wrapFailure(mlir::LogicalResult result, std::string message) {
  if (mlir::failed(result))
    std::cout << message << " failed" << std::endl;
}

extern "C" {

void *oclInit() {
  std::cout << "init" << std::endl;
  return new OpenClRuntime();
}

void oclDeinit(void *runtime) {
  std::cout << "deinit" << std::endl;
  delete reinterpret_cast<OpenClRuntime *>(runtime);
}

void *oclCreateKernel(void *runtime, uint8_t *spirv, uint32_t spirv_size) {
  std::cout << "kernel" << std::endl;
  auto result = reinterpret_cast<OpenClRuntime *>(runtime)->createKernel(
      spirv, spirv_size);
  return wrapFailure(result, "oclCreateKernel");
}

void *oclEnqueueKernel(void *runtime, void *kernel, uint64_t dimX,
                       uint64_t dimY, uint64_t dimZ, uint64_t localX,
                       uint64_t localY, uint64_t localZ, void *dep) {
  std::cout << "enqueue" << std::endl;
  auto result = reinterpret_cast<OpenClRuntime *>(runtime)->enqueueKernel(
      reinterpret_cast<OpenClKernel *>(kernel), dimX, dimY, dimZ, localX,
      localY, localZ, reinterpret_cast<OpenClEvent *>(dep));
  return wrapFailure(result, "oclEnqueueKernel");
}

void oclSetKernelArg(void *runtime, void *kernel, uint32_t idx, void *buffer) {
  std::cout << "arg" << std::endl;
  auto result = reinterpret_cast<OpenClRuntime *>(runtime)->setBufferArg(
      reinterpret_cast<OpenClKernel *>(kernel), idx,
      reinterpret_cast<OpenClBuffer *>(buffer));
  return wrapFailure(result, "oclSetKernelArg");
}

void *oclAllocBuffer(void *runtime, uint32_t length, void *ptr) {
  std::cout << "alloc" << std::endl;
  auto result =
      reinterpret_cast<OpenClRuntime *>(runtime)->allocBuffer(ptr, length);
  return wrapFailure(result, "oclAllocBuffer");
}

void oclDeallocBuffer(void *buffer) {
  std::cout << "dealloc" << std::endl;
  auto result = deallocBuffer(reinterpret_cast<OpenClBuffer *>(buffer));
  return wrapFailure(result, "oclDeallocBuffer");
}

void *oclEnqueueRead(void *runtime, void *buffer, void *ptr, void *dep) {
  std::cout << "read" << std::endl;
  auto result = reinterpret_cast<OpenClRuntime *>(runtime)->enqueueRead(
      reinterpret_cast<OpenClBuffer *>(buffer), ptr,
      reinterpret_cast<OpenClEvent *>(dep));
  return wrapFailure(result, "oclEnqueueRead");
}

void *oclEnqueueWrite(void *runtime, void *buffer, void *ptr, void *dep) {
  std::cout << "write" << std::endl;
  auto result = reinterpret_cast<OpenClRuntime *>(runtime)->enqueueWrite(
      reinterpret_cast<OpenClBuffer *>(buffer), ptr,
      reinterpret_cast<OpenClEvent *>(dep));
  return wrapFailure(result, "oclEnqueueWrite");
}

void oclFlush(void *runtime) {
  std::cout << "flush" << std::endl;
  auto result = reinterpret_cast<OpenClRuntime *>(runtime)->flush();
  return wrapFailure(result, "oclFlush");
}

void *oclGroupEvents(uint32_t count, ...) {
  std::cout << "group" << std::endl;
  va_list args;
  va_start(args, count);
  std::vector<OpenClEvent *> events;
  for (uint32_t i = 0; i < count; ++i)
    events.push_back(reinterpret_cast<OpenClEvent *>(va_arg(args, void *)));
  va_end(args);
  auto result = groupEvents(events);
  return wrapFailure(result, "oclGroupEvents");
}

void oclWait(void *event) {
  std::cout << "wait" << std::endl;
  auto result = wait(reinterpret_cast<OpenClEvent *>(event));
  return wrapFailure(result, "oclWait");
}

} // extern "C"

namespace {
struct Registration {
  Registration() {
    using pmlc::compiler::registerSymbol;
    std::cout << "Register" << std::endl;

    // OpenCL Runtime functions
    registerSymbol("oclInit", reinterpret_cast<void *>(oclInit));
    registerSymbol("oclDeinit", reinterpret_cast<void *>(oclDeinit));
    registerSymbol("oclCreateKernel",
                   reinterpret_cast<void *>(oclCreateKernel));
    registerSymbol("oclEnqueueKernel",
                   reinterpret_cast<void *>(oclEnqueueKernel));
    registerSymbol("oclSetKernelArg",
                   reinterpret_cast<void *>(oclSetKernelArg));
    registerSymbol("oclAllocBuffer", reinterpret_cast<void *>(oclAllocBuffer));
    registerSymbol("oclDeallocBuffer",
                   reinterpret_cast<void *>(oclDeallocBuffer));
    registerSymbol("oclEnqueueRead", reinterpret_cast<void *>(oclEnqueueRead));
    registerSymbol("oclEnqueueWrite",
                   reinterpret_cast<void *>(oclEnqueueWrite));
    registerSymbol("oclGroupEvents", reinterpret_cast<void *>(oclGroupEvents));
    registerSymbol("oclWait", reinterpret_cast<void *>(oclWait));
  }
};
static Registration reg;
} // namespace
