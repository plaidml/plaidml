// Copyright 2020 Intel Corporation

#include "pmlc/rt/opencl/opencl_runtime.h"

#include <string>
#include <vector>

#include <iostream>

namespace oclrt {

struct OpenClKernel {
  cl::Kernel kernel;
};

struct OpenClBuffer {
  cl::Buffer buffer;
  uint32_t length;
};

struct OpenClEvent {
  std::vector<cl::Event> events;
  OpenClRuntime *runtime;
};

namespace {

cl::Device createDevice() {
  std::vector<cl::Platform> platforms;
  std::vector<cl::Device> devices;
  cl::Platform::get(&platforms);
  for (auto &p : platforms) {
    auto vendor = p.getInfo<CL_PLATFORM_VENDOR>();
    if (vendor.find("Intel") != std::string::npos) {
      p.getDevices(CL_DEVICE_TYPE_GPU, &devices);
      if (!devices.empty())
        return devices.front();
      devices.clear();
    }
  }
  // TODO Fix this
  return cl::Device();
}

} // namespace

OpenClRuntime::OpenClRuntime()
    : rtDevice(createDevice()), rtContext(rtDevice), rtQueue(rtContext) {}

OpenClRuntime::~OpenClRuntime() { rtQueue.finish(); }

mlir::FailureOr<OpenClKernel *> OpenClRuntime::createKernel(uint8_t *spirv,
                                                            uint32_t length) {
  cl_int err = CL_SUCCESS;
  auto clProgram = clCreateProgramWithIL(rtContext.get(), spirv, length, &err);
  if (CL_SUCCESS != err)
    return mlir::failure();
  auto program = cl::Program(clProgram);
  if (CL_SUCCESS != program.build({rtDevice})) {
    auto log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
    for (auto &l : log) {
      std::cout << l.second << std::endl;
    }
    return mlir::failure();
  }

  auto log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
  for (auto &l : log) {
    std::cout << l.second << std::endl;
  }

  std::vector<cl::Kernel> kernels;

  if (CL_SUCCESS != program.createKernels(&kernels) || kernels.size() != 1) {
    std::cout << "kernels " << kernels.size() << std::endl;
    return mlir::failure();
  }

  rtKernels.emplace_back(new OpenClKernel{kernels[0]});
  return rtKernels.back().get();
}

mlir::FailureOr<OpenClEvent *>
OpenClRuntime::enqueueKernel(OpenClKernel *kernel, uint32_t dimX, uint32_t dimY,
                             uint32_t dimZ, uint32_t localX, uint32_t localY,
                             uint32_t localZ, OpenClEvent *dep) {
  if (!kernel)
    return mlir::failure();
  auto clKernel = kernel->kernel;

  auto offset = cl::NDRange();
  auto global = cl::NDRange(dimX, dimY, dimZ);
  auto local = cl::NDRange(localX, localY, localZ);
  cl::Event ev;
  std::vector<cl::Event> dependencies;
  if (dep)
    dependencies = dep->events;

  auto status = rtQueue.enqueueNDRangeKernel(clKernel, offset, global, local,
                                             &dependencies, &ev);
  if (CL_SUCCESS != status) {
    std::cout << "status: " << status << std::endl;
    std::cout << "global: " << dimX << " " << dimY << " " << dimZ << std::endl;
    std::cout << "local: " << localX << " " << localY << " " << localZ
              << std::endl;
    return mlir::failure();
  }

  auto wrappedEvent = new OpenClEvent();
  wrappedEvent->events.push_back(ev);
  wrappedEvent->runtime = this;
  rtEvents.emplace_back(wrappedEvent);
  return wrappedEvent;
}

mlir::LogicalResult OpenClRuntime::setBufferArg(OpenClKernel *kernel,
                                                int32_t idx,
                                                OpenClBuffer *buffer) {
  if (!kernel || !buffer)
    return mlir::failure();

  auto status = kernel->kernel.setArg(idx, buffer->buffer);
  return mlir::success(CL_SUCCESS == status);
}

mlir::FailureOr<OpenClBuffer *> OpenClRuntime::allocBuffer(void *ptr,
                                                           int32_t length) {
  cl_int err = CL_SUCCESS;
  cl_mem_flags flags = CL_MEM_READ_WRITE;
  if (ptr)
    flags = flags | CL_MEM_COPY_HOST_PTR;
  std::cout << flags << std::endl;
  auto buffer = cl::Buffer(rtContext, flags, (size_t)length, ptr, &err);
  if (CL_SUCCESS != err) {
    std::cout << "status: " << err << ", " << length << std::endl;
    return mlir::failure();
  }
  std::cout << "allocated" << std::endl;

  auto rtBuffer = new OpenClBuffer();
  rtBuffer->buffer = buffer;
  rtBuffer->length = length;
  return rtBuffer;
}

mlir::FailureOr<OpenClEvent *>
OpenClRuntime::enqueueRead(OpenClBuffer *buffer, void *ptr, OpenClEvent *dep) {
  if (!buffer || !ptr)
    return mlir::failure();

  cl::Event ev;
  std::vector<cl::Event> dependencies;
  if (dep)
    dependencies = dep->events;
  // cl_int err;
  // auto mapPtr = rtQueue.enqueueMapBuffer(buffer->buffer, true, CL_MAP_READ |
  // CL_MAP_WRITE, 0, buffer->length, &dependencies, nullptr, &err); if
  // (CL_SUCCESS != err) {
  //   std::cout << "map failed: " << err << std::endl;
  //   return mlir::failure();
  // }
  // std::cout << "mapped " << buffer->length << std::endl;
  // // for (size_t i = 0; i < buffer->length; ++i) {
  // //   std::cout << i << std::endl;
  // //   static_cast<char*>(ptr)[i] = static_cast<char*>(mapPtr)[i];
  // // }
  // std::cout << "copied" << std::endl;
  // err = rtQueue.enqueueUnmapMemObject(buffer->buffer, mapPtr, nullptr, &ev);
  // if (CL_SUCCESS != err) {
  //   std::cout << "unmap failed: " << err << std::endl;
  //   return mlir::failure();
  // }
  //   std::cout << "unmapped" << std::endl;

  //
  auto err = rtQueue.enqueueReadBuffer(buffer->buffer, false, 0, buffer->length,
                                       ptr, &dependencies, &ev);
  if (CL_SUCCESS != err) {
    std::cout << "status: " << err << std::endl;
    return mlir::failure();
  }

  auto rtEvent = new OpenClEvent();
  rtEvent->events.push_back(ev);
  rtEvent->runtime = this;
  rtEvents.emplace_back(rtEvent);
  return rtEvent;
}

mlir::FailureOr<OpenClEvent *>
OpenClRuntime::enqueueWrite(OpenClBuffer *buffer, void *ptr, OpenClEvent *dep) {
  if (!buffer)
    return mlir::failure();

  cl::Event ev;
  std::vector<cl::Event> dependencies;
  if (dep)
    dependencies = dep->events;

  auto status = rtQueue.enqueueWriteBuffer(
      buffer->buffer, false, 0, buffer->length, ptr, &dependencies, &ev);
  if (CL_SUCCESS != status)
    return mlir::failure();

  auto rtEvent = new OpenClEvent();
  rtEvent->events.push_back(ev);
  rtEvent->runtime = this;
  rtEvents.emplace_back(rtEvent);
  return rtEvent;
}

mlir::LogicalResult OpenClRuntime::flush() {
  return mlir::success(CL_SUCCESS == rtQueue.flush());
}

void OpenClRuntime::registerEvent(OpenClEvent *event) {
  rtEvents.emplace_back(event);
}

mlir::LogicalResult deallocBuffer(OpenClBuffer *ptr) {
  if (!ptr)
    return mlir::failure();
  delete ptr;
  return mlir::success();
}

mlir::FailureOr<OpenClEvent *> groupEvents(std::vector<OpenClEvent *> events) {
  if (events.empty())
    return mlir::failure();

  std::vector<cl::Event> grouped;
  for (auto &rtEvent : events) {
    if (!rtEvent)
      continue; // Should it be failure?
    for (auto &ev : rtEvent->events) {
      grouped.push_back(ev);
    }
  }
  OpenClRuntime *runtime;
  runtime = events[0]->runtime;

  auto rtEvent = new OpenClEvent();
  rtEvent->events = grouped;
  rtEvent->runtime = runtime;
  runtime->registerEvent(rtEvent);
  return rtEvent;
}

mlir::LogicalResult wait(OpenClEvent *event) {
  if (!event)
    return mlir::failure();

  for (auto &ev : event->events) {
    ev.wait();
  }
  return mlir::success();
}

} // namespace oclrt
