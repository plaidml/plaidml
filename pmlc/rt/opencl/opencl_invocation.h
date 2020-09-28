// Copyright 2020 Intel Corporation
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "pmlc/rt/opencl/opencl_device.h"

namespace pmlc::rt::opencl {
class OpenCLEvent;

/// Class encapsulating OpenCL memory buffer allocated on device.
class OpenCLMemory {
public:
  OpenCLMemory(cl::Buffer buffer, size_t bytes)
      : buffer(buffer), bytes(bytes) {}

  /// Returns OpenCL buffer.
  cl::Buffer getBuffer() { return buffer; }
  /// Returns size of buffer in bytes.
  size_t size() { return bytes; }
  /// Enqueues read operation from this buffer into `dst` pointer
  /// on specified command queue.
  cl::Event enqueueRead(cl::CommandQueue queue, void *dst,
                        const std::vector<cl::Event> &dependencies);
  /// Enqueues write operation from `src` pointer into this buffer
  /// on specified command queue.
  cl::Event enqueueWrite(cl::CommandQueue queue, void *src,
                         const std::vector<cl::Event> &dependencies);

private:
  cl::Buffer buffer;
  size_t bytes;
};

/// OpenCL kernel with additional state information:
/// name and event dependencies.
/// Serves as one time incremental object, `enqueue()` should be
/// called only once and does not reset it's internal state.
class OpenCLKernel {
public:
  /// Constructs kernel declared with `name` from compiled `program`.
  OpenCLKernel(cl::Program program, std::string name);

  /// Adds event dependency that must be completed before this kernel.
  void addDependency(OpenCLEvent *event);
  /// Sets kernel argument `idx` to `memory`.
  void setArg(unsigned idx, OpenCLMemory *memory);
  /// Enqueues wrapped kernel on specified command queue `queue` with
  /// `gws` global work size and `lws` local work size.
  /// Returns OpenCL event tracking execution of kernel execution.
  cl::Event enqueue(cl::CommandQueue queue, cl::NDRange gws, cl::NDRange lws);
  /// Returns name of this kernel.
  const std::string &getName() const { return name; }

private:
  cl::Kernel kernel;
  std::string name;
  std::vector<cl::Event> dependencies;
};

/// Kind of asynchronous operation executed on OpenCL device.
enum class OpenCLActionKind { Barrier, Kernel, Read, Write };

/// Class encapsulating OpenCL event that describes status of
/// operation that produced it and serves for ordering operations.
class OpenCLEvent {
public:
  OpenCLEvent(cl::Event event, OpenCLActionKind kind, std::string name);

  /// Returns OpenCL event object.
  cl::Event getEvent() const { return event; }
  /// Returns kind of operation that this event describes.
  OpenCLActionKind getKind() const { return kind; }
  /// Returns name of operation that this event describes.
  const std::string &getName() const { return name; }

  /// Blocks execution untill all `events` have finished executing.
  static void wait(const std::vector<OpenCLEvent *> &events);

private:
  cl::Event event;
  OpenCLActionKind kind;
  std::string name;
};

// OpenCLInvocation encapsulates a particular run of a network on a OpenCL
// device. It's instantiated and managed from the JITted network code, using
// callbacks in opencl_wrappers.cc.
class OpenCLInvocation {
public:
  OpenCLInvocation();
  ~OpenCLInvocation();

  /// Allocates memory on OpenCL device with specified size in bytes.
  /// If data is not NULL, then its content is copied to allocated memory.
  OpenCLMemory *allocateMemory(size_t bytes, void *data = NULL);
  /// Releases memory obtained from `allocateMemory` call.
  /// Any further use of `memory` is invalid.
  void deallocateMemory(OpenCLMemory *memory);
  /// Enqueues operation to read data from `src` memory into `dst` pointer.
  /// Read will start executing only after operations connected to `deps`
  /// events are finished.
  /// Returns event describing status of this operation.
  OpenCLEvent *enqueueRead(OpenCLMemory *src, void *dst,
                           const std::vector<OpenCLEvent *> &deps);
  /// Enqueues operation to write data from `src` pointer into `dst` memory.
  /// Write will start executing only after operations connected to `deps`
  /// events are finished.
  /// Returns event describing status of this operation.
  OpenCLEvent *enqueueWrite(OpenCLMemory *dst, void *src,
                            const std::vector<OpenCLEvent *> &deps);
  /// Creates OpenCL kernel from SPIRV binary.
  OpenCLKernel *createKernelFromIL(char *data, size_t bytes, const char *name);
  /// Enqueues kernel execution on device with specified global and local work
  /// sizes. Kernel object is destroyed after this function finishes. Returns
  /// event describing status of kernel execution.
  OpenCLEvent *enqueueKernel(OpenCLKernel *kernel, cl::NDRange gws,
                             cl::NDRange lws);
  /// Enqueues barrier operation on device. Barrier serves as synchronization
  /// primitive that ensures all operations enqueued after it finishes will
  /// start executing only after all operations before have finished.
  /// Barrier goes into effect after dependant events `deps` are finished
  /// or immiediately if `deps` is empty.
  /// Returns event describing status of this barrier operation.
  OpenCLEvent *enqueueBarrier(const std::vector<OpenCLEvent *> &deps);
  /// Forces starting execution of previosuly enqueued operations that aren't
  /// blocked by any dependencies.
  void flush();
  /// Blocks untill all previously enqueued operations are finished.
  void finish();

private:
  std::shared_ptr<OpenCLDevice> device;
  OpenCLQueueUser queueUser;
  std::vector<std::unique_ptr<OpenCLEvent>> events;

  OpenCLEvent *wrapEvent(cl::Event event, OpenCLActionKind kind,
                         std::string name);
};

} // namespace pmlc::rt::opencl
