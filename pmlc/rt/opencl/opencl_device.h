// Copyright 2020 Intel Corporation
#pragma once

#include <atomic>
#include <memory>
#include <vector>

#include "pmlc/rt/opencl/opencl_utils.h"
#include "pmlc/rt/runtime.h"

namespace pmlc::rt::opencl {

/// OpenCL command queue wrapper.
class OpenCLQueue {
public:
  OpenCLQueue(const cl::Context &context, const cl::Device &device,
              cl::QueueProperties properties);

  /// Returns wrapped OpenCL queue.
  cl::CommandQueue getOclQueue() { return queue; }
  /// Returns properties wrapped OpenCL queue was created with.
  cl::QueueProperties getOclProperties() { return properties; }

private:
  cl::CommandQueue queue;
  cl::QueueProperties properties;
};

class OpenCLQueueGuard;

/// Class providing RAII idiomatic way for reserving OpenCLQueue for use.
/// This class is not intended to be created manually, instead instance
/// should be obtained from OpenCLDevice.
/// Upon destruction this queue will be released and made availiable for
/// future reuse.
class OpenCLQueueUser {
public:
  /// Creates null user not referencing any OpenCL queue.
  OpenCLQueueUser();
  /// Deleted copy constructor.
  OpenCLQueueUser(const OpenCLQueueUser &copy) = delete;
  /// Custom move constructor.
  OpenCLQueueUser(OpenCLQueueUser &&move);
  /// Custom destructor - releases use from OpenCLQueueGuard.
  ~OpenCLQueueUser();
  /// Deleted copy assignment.
  OpenCLQueueUser &operator=(const OpenCLQueueUser &) = delete;

  /// Returns OpenCL queue this user has reserved.
  cl::CommandQueue getOclQueue() { return queue->getOclQueue(); }
  /// Returns properties of reserved queue.
  cl::QueueProperties getOclProperties() { return queue->getOclProperties(); }
  /// Returns true if this is non-empty user.
  operator bool() const { return nullptr != guard; }

private:
  OpenCLQueueUser(OpenCLQueueGuard *guard, OpenCLQueue *queue);

  OpenCLQueueGuard *guard;
  OpenCLQueue *queue;
  // Friendship definition needed to access private constructor.
  friend class OpenCLQueueGuard;
};

/// Class guarding access to OpenCLQueue that can have only one user,
/// represented as OpenCLQueueUser.
class OpenCLQueueGuard {
public:
  explicit OpenCLQueueGuard(OpenCLQueue queue, bool init = false)
      : used(init), queue(queue) {}

  /// Returns properties of contained queue.
  cl::QueueProperties getOclProperties() { return queue.getOclProperties(); }
  /// Checks if this queue is currently in use. Does not guarantee that
  /// subsequent calls to use() will return non-empty user in
  /// multithreaded scenerios.
  bool isUsed();
  /// Tries to reserve queue for use and returns corresponding user.
  /// If queue is currently in use and cannot be reserved returns
  /// empty user.
  OpenCLQueueUser use();

private:
  std::atomic<bool> used;
  OpenCLQueue queue;
  // Fiendship definition to release guard when upon user destruction.
  friend class OpenCLQueueUser;
};

/// OpenCL device abstraction.
/// Purpose of this class is to manage OpenCL resources connected
/// to device.
class OpenCLDevice final : public pmlc::rt::Device {
public:
  explicit OpenCLDevice(cl::Device device);

  /// Returns OpenCL context created with only this device.
  cl::Context getOclContext() { return context; }
  /// Returns OpenCL device.
  cl::Device getOclDevice() { return device; }
  /// Returns OpenCL queue with specified properties for execution
  /// on this device. Returned user is always non-empty.
  OpenCLQueueUser getQueue(cl::QueueProperties properties);

private:
  cl::Context context;
  cl::Device device;
  std::vector<std::unique_ptr<OpenCLQueueGuard>> queues;
  std::mutex queuesMutex;
};

} // namespace pmlc::rt::opencl
