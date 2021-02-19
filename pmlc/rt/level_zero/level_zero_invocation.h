// Copyright 2021 Intel Corporation
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "pmlc/rt/level_zero/level_zero_device.h"

namespace pmlc::rt::level_zero {
class LevelZeroInvocation;
class LevelZeroEvent;

/// Kind of memory type.
enum class LevelZeroMemoryKind { Host, Device, Shared };

/// Class encapsulating LevelZero memory buffer allocated on device.
class LevelZeroMemory {
public:
  LevelZeroMemory(void *buffer, size_t bytes, LevelZeroMemoryKind kind,
                  ze_context_handle_t context)
      : buffer(buffer), bytes(bytes), kind(kind), context(context) {}
  ~LevelZeroMemory() { lzu::free_memory(context, buffer); }

  /// Returns allocated buffer.
  void *getBuffer() { return buffer; }
  /// Returns size of buffer in bytes.
  size_t size() { return bytes; }
  /// Returns kind of memory.
  LevelZeroMemoryKind getKind() const { return kind; }
  /// Enqueues read operation from this buffer into `dst` pointer
  /// on specified command queue.
  void enqueueRead(ze_command_list_handle_t list, void *dst,
                   std::vector<ze_event_handle_t> &dependencies,
                   ze_event_handle_t &resultE);
  /// Enqueues write operation from `src` pointer into this buffer
  /// on specified command queue.
  void enqueueWrite(ze_command_list_handle_t list, void *src,
                    std::vector<ze_event_handle_t> &dependencies,
                    ze_event_handle_t &resultE);

private:
  void *buffer;
  size_t bytes;
  LevelZeroMemoryKind kind;
  ze_context_handle_t context;
};

/// LevelZero kernel wrapper
class LevelZeroKernel {
public:
  /// Constructs kernel declared with `name` from compiled `module`.
  LevelZeroKernel(ze_module_handle_t module, std::string name);
  ~LevelZeroKernel();

  /// Adds event dependency that must be completed before this kernel.
  void addDependency(LevelZeroEvent *event);
  /// Sets kernel argument `idx` to `memory`.
  void setArg(unsigned idx, LevelZeroMemory *memory);
  /// Enqueues wrapped kernel on specified command queue `queue` with
  /// `gws` global work size and `lws` local work size.
  /// The gws and lws shall be converted for LevelZero API.
  /// Returns LevelZero event tracking execution of kernel execution.
  void enqueue(ze_command_list_handle_t list, ze_group_count_t gws,
               ze_group_count_t lws, ze_event_handle_t &resultE);
  /// Returns name of this kernel.
  const std::string &getName() const { return name; }

  ze_kernel_handle_t getKernel() { return kernel; }

private:
  ze_module_handle_t module;
  ze_kernel_handle_t kernel;
  std::string name;
  std::vector<ze_event_handle_t> dependencies;
};

/// Kind of asynchronous operation executed on LevelZero device.
enum class LevelZeroActionKind { Barrier, Kernel, Read, Write };

/// Class encapsulating LevelZero event that describes status of
/// operation that produced it and serves for ordering operations.
class LevelZeroEvent {
public:
  LevelZeroEvent(LevelZeroInvocation *invocation, ze_event_handle_t event,
                 LevelZeroActionKind kind, std::string name);

  /// Returns LevelZero event object.
  ze_event_handle_t getEvent() const { return event; }
  /// Returns kind of operation that this event describes.
  LevelZeroActionKind getKind() const { return kind; }
  /// Returns name of operation that this event describes.
  const std::string &getName() const { return name; }
  /// Return timestamp that event has taken.
  ze_kernel_timestamp_result_t getTimestamp() const { return timestamp; }
  /// Get invocation pointer.
  LevelZeroInvocation *getInvocation() const { return invocation; }

  /// Clear event after it has been consumed ( waited ).
  void setEvent(ze_event_handle_t e) { event = e; }
  /// Set timestamp.
  void setTimestamp(ze_kernel_timestamp_result_t value) { timestamp = value; }

  /// Blocks execution until all `events` have finished executing.
  static void wait(const std::vector<LevelZeroEvent *> &events);

private:
  LevelZeroInvocation *invocation;
  ze_event_handle_t event;
  LevelZeroActionKind kind;
  std::string name;
  ze_kernel_timestamp_result_t timestamp;
};

// LevelZeroInvocation encapsulates a particular run of a network on a LevelZero
// device. It's instantiated and managed from the JITted network code, using
// callbacks in level_zero_wrappers.cc.
class LevelZeroInvocation {
public:
  explicit LevelZeroInvocation(LevelZeroDevice *device,
                               ze_command_queue_group_properties_t prop);
  ~LevelZeroInvocation();

  /// Allocates memory on LevelZero device with specified size in bytes.
  LevelZeroMemory *allocateMemory(size_t bytes, LevelZeroMemoryKind kind);
  /// Releases memory obtained from `allocateMemory` call.
  /// But as kernel may not really run here, may just do record.
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
  /// Creates LevelZero kernel from SPIRV binary.
  LevelZeroKernel *createKernelFromIL(char *data, size_t bytes,
                                      const char *name);
  /// Enqueues kernel execution on device with specified global and local work
  /// sizes. Kernel object needs to be destroyed manually. Returns
  /// event describing status of kernel execution.
  LevelZeroEvent *enqueueKernel(LevelZeroKernel *kernel, ze_group_count_t gws,
                                ze_group_count_t lws);
  /// Not used now.
  LevelZeroEvent *enqueueBarrier(const std::vector<LevelZeroEvent *> &deps);
  /// Release ze_event_handle_t but not LevelZeroEvent as it may keep profiling
  /// info.
  void releaseZeEvent(ze_event_handle_t event);
  /// Forces starting execution of previosuly enqueued operations that aren't
  /// blocked by any dependencies.
  void flush();
  /// Blocks untill all previously enqueued operations are finished.
  void finish();

private:
  std::shared_ptr<LevelZeroDevice> device;
  LevelZeroQueueUser queueUser;
  lzu::zeEventPool eventPool;
  // Resource clear lists
  std::vector<std::unique_ptr<LevelZeroEvent>> events;
  std::vector<LevelZeroMemory *> memories;

  LevelZeroEvent *wrapEvent(ze_event_handle_t event, LevelZeroActionKind kind,
                            std::string name);
};

} // namespace pmlc::rt::level_zero
