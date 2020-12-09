// Copyright 2020 Intel Corporation
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "pmlc/rt/level_zero/level_zero_device.h"

namespace pmlc::rt::level_zero {
class LevelZeroInvocation;
class LevelZeroEvent;

/// Class encapsulating level zero memory buffer allocated on device.
class LevelZeroMemory {
public:
  LevelZeroMemory(void *buffer, size_t bytes) : buffer(buffer), bytes(bytes) {}
  ~LevelZeroMemory() {lzt::free_memory(buffer);}

  /// Returns OpenCL buffer.
  void *getBuffer() { return buffer; }
  /// Returns size of buffer in bytes.
  size_t size() { return bytes; }
  /// Enqueues read operation from this buffer into `dst` pointer
  /// on specified command queue.
  void
  enqueueRead(ze_command_list_handle_t list, void *dst,
              std::vector<ze_event_handle_t> &dependencies, ze_event_handle_t &resultE);
  /// Enqueues write operation from `src` pointer into this buffer
  /// on specified command queue.
  void
  enqueueWrite(ze_command_list_handle_t list, void *src,
               std::vector<ze_event_handle_t> &dependencies, ze_event_handle_t &resultE);

private:
  void *buffer;
  size_t bytes;
};

/// OpenCL kernel with additional state information:
/// name and event dependencies.
/// Serves as one time incremental object, `enqueue()` should be
/// called only once and does not reset it's internal state.
class LevelZeroKernel {
public:
  /// Constructs kernel declared with `name` from compiled `program`.
  LevelZeroKernel(ze_module_handle_t module, std::string name);
  ~LevelZeroKernel();

  /// Adds event dependency that must be completed before this kernel.
  void addDependency(LevelZeroEvent* event);
  /// Sets kernel argument `idx` to `memory`.
  void setArg(unsigned idx, LevelZeroMemory *memory);
  /// Enqueues wrapped kernel on specified command queue `queue` with
  /// `gws` global work size and `lws` local work size.
  /// Returns OpenCL event tracking execution of kernel execution.
  void enqueue(ze_command_list_handle_t list, ze_group_count_t gws,
                            ze_group_count_t lws, ze_event_handle_t &resultE);
  /// Returns name of this kernel.
  const std::string &getName() const { return name; }

private:
  ze_module_handle_t module;
  ze_kernel_handle_t kernel;
  std::string name;
  std::vector<ze_event_handle_t> dependencies;
};

/// Kind of asynchronous operation executed on OpenCL device.
enum class LevelZeroActionKind { Barrier, Kernel, Read, Write };

/// Class encapsulating OpenCL event that describes status of
/// operation that produced it and serves for ordering operations.
class LevelZeroEvent {
public:
  LevelZeroEvent(ze_event_handle_t event, LevelZeroActionKind kind,
                 std::string name);

  /// Returns OpenCL event object.
  ze_event_handle_t getEvent() const { return event; }
  /// Returns kind of operation that this event describes.
  LevelZeroActionKind getKind() const { return kind; }
  /// Returns name of operation that this event describes.
  const std::string &getName() const { return name; }

  /// Blocks execution untill all `events` have finished executing.
  static void wait(const std::vector<LevelZeroEvent* > &events);

private:
  ze_event_handle_t event;
  LevelZeroActionKind kind;
  std::string name;
};

// OpenCLInvocation encapsulates a particular run of a network on a OpenCL
// device. It's instantiated and managed from the JITted network code, using
// callbacks in opencl_wrappers.cc.
class LevelZeroInvocation {
public:
  explicit LevelZeroInvocation(LevelZeroDevice *device);
  ~LevelZeroInvocation();

  /// Allocates memory on OpenCL device with specified size in bytes.
  LevelZeroMemory *allocateMemory(size_t bytes);
  /// Releases memory obtained from `allocateMemory` call.
  /// Any further use of `memory` is invalid.
  void deallocateMemory(LevelZeroMemory *memory);
  /// Enqueues operation to read data from `src` memory into `dst` pointer.
  /// Read will start executing only after operations connected to `deps`
  /// events are finished.
  /// Returns event describing status of this operation.
  LevelZeroEvent *enqueueRead(LevelZeroMemory *src, void *dst,
                              const std::vector<LevelZeroEvent *> &deps);
  /// Enqueues operation to write data from `src` pointer into `dst` memory.
  /// Write will start executing only after operations connected to `deps`
  /// events are finished.
  /// Returns event describing status of this operation.
  LevelZeroEvent *enqueueWrite(LevelZeroMemory *dst, void *src,
                               const std::vector<LevelZeroEvent *> &deps);
  /// Creates OpenCL kernel from SPIRV binary.
  LevelZeroKernel *createKernelFromIL(char *data, size_t bytes,
                                      const char *name);
  /// Enqueues kernel execution on device with specified global and local work
  /// sizes. Kernel object is destroyed after this function finishes. Returns
  /// event describing status of kernel execution.
  LevelZeroEvent *enqueueKernel(LevelZeroKernel *kernel, ze_group_count_t gws,
                                ze_group_count_t lws);
  /// Enqueues barrier operation on device. Barrier serves as synchronization
  /// primitive that ensures all operations enqueued after it finishes will
  /// start executing only after all operations before have finished.
  /// Barrier goes into effect after dependant events `deps` are finished
  /// or immiediately if `deps` is empty.
  /// Returns event describing status of this barrier operation.
  LevelZeroEvent *enqueueBarrier(const std::vector<LevelZeroEvent *> &deps);
  /// Forces starting execution of previosuly enqueued operations that aren't
  /// blocked by any dependencies.
  void flush();
  /// Blocks untill all previously enqueued operations are finished.
  void finish();

private:
  std::shared_ptr<LevelZeroDevice> device;
  LevelZeroQueueUser queueUser;
  level_zero_tests::zeEventPool eventPool;
  // clear list
  std::vector<std::unique_ptr<LevelZeroEvent>> events;
  std::vector<LevelZeroKernel *> kernels;
  std::vector<LevelZeroMemory *> memories;

  LevelZeroEvent *wrapEvent(ze_event_handle_t event, LevelZeroActionKind kind,
                            std::string name);
};

} // namespace pmlc::rt::level_zero
