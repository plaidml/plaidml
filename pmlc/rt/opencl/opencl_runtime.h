// Copyright 2020 Intel Corporation

#pragma once

#include <memory>
#include <vector>

#include "CL/cl2.hpp"

#include "mlir/Support/LogicalResult.h"

namespace oclrt {

struct OpenClKernel;
struct OpenClEvent;
struct OpenClBuffer;

class OpenClRuntime {
public:
  OpenClRuntime();
  virtual ~OpenClRuntime();

  mlir::FailureOr<OpenClKernel *> createKernel(uint8_t *spirv, uint32_t length);
  mlir::FailureOr<OpenClEvent *> enqueueKernel(OpenClKernel *kernel,
                                               uint32_t dimX, uint32_t dimY,
                                               uint32_t dimZ, uint32_t localX,
                                               uint32_t localY, uint32_t localZ,
                                               OpenClEvent *dep);
  mlir::LogicalResult setBufferArg(OpenClKernel *kernel, int32_t idx,
                                   OpenClBuffer *buffer);

  mlir::FailureOr<OpenClBuffer *> allocBuffer(void *ptr, int32_t length);
  mlir::FailureOr<OpenClEvent *> enqueueRead(OpenClBuffer *buf, void *ptr,
                                             OpenClEvent *dep);
  mlir::FailureOr<OpenClEvent *> enqueueWrite(OpenClBuffer *buf, void *ptr,
                                              OpenClEvent *dep);

  mlir::LogicalResult flush();

  void registerEvent(OpenClEvent *event);

private:
  cl::Device rtDevice;
  cl::Context rtContext;
  cl::CommandQueue rtQueue;
  std::vector<std::unique_ptr<OpenClKernel>> rtKernels;
  std::vector<std::unique_ptr<OpenClEvent>> rtEvents;
};

mlir::LogicalResult deallocBuffer(OpenClBuffer *ptr);
mlir::FailureOr<OpenClEvent *> groupEvents(std::vector<OpenClEvent *> events);
mlir::LogicalResult wait(OpenClEvent *event);

} // namespace oclrt
