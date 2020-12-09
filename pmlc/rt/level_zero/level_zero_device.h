// Copyright 2020 Intel Corporation
#pragma once

#include <atomic>
#include <memory>
#include <vector>

#include "pmlc/rt/level_zero/level_zero_utils.h"
#include "pmlc/rt/runtime.h"

namespace pmlc::rt::level_zero {

/// OpenCL command queue wrapper.
class LevelZeroQueue {
public:
  LevelZeroQueue(const ze_context_handle_t &context,
                 const ze_device_handle_t &device,
                 ze_command_queue_group_properties_t properties);
  ~LevelZeroQueue();

  /// Returns wrapped level_zero queue.
  ze_command_queue_handle_t getLevelZeroQueue() { return queue; }
  ze_command_list_handle_t getLevelZeroList() { return list; }
  /// Returns properties wrapped OpenCL queue was created with.
  ze_command_queue_group_properties_t getLevelZeroProperties() {
    return properties;
  }

private:
  ze_command_queue_handle_t queue;
  ze_command_list_handle_t list;
  ze_command_queue_group_properties_t properties;
};

class LevelZeroQueueGuard;

/// Class providing RAII idiomatic way for reserving OpenCLQueue for use.
/// This class is not intended to be created manually, instead instance
/// should be obtained from OpenCLDevice.
/// Upon destruction this queue will be released and made availiable for
/// future reuse.
class LevelZeroQueueUser {
public:
  /// Creates null user not referencing any level zero queue.
  LevelZeroQueueUser();
  /// Deleted copy constructor.
  LevelZeroQueueUser(const LevelZeroQueueUser &copy) = delete;
  /// Custom move constructor.
  LevelZeroQueueUser(LevelZeroQueueUser &&move);
  /// Custom destructor - releases use from LevelZeroQueueGuard.
  ~LevelZeroQueueUser();
  /// Deleted copy assignment.
  LevelZeroQueueUser &operator=(const LevelZeroQueueUser &) = delete;

  /// Returns level zero queue this user has reserved.
  ze_command_queue_handle_t getLevelZeroQueue() {
    return queue->getLevelZeroQueue();
  }
  ze_command_list_handle_t getLevelZeroList() {
    return queue->getLevelZeroList();
  }
  /// Returns properties of reserved queue.
  ze_command_queue_group_properties_t getLevelZeroProperties() {
    return queue->getLevelZeroProperties();
  }
  /// Returns true if this is non-empty user.
  operator bool() const { return nullptr != guard; }

private:
  LevelZeroQueueUser(LevelZeroQueueGuard *guard, LevelZeroQueue *queue);

  LevelZeroQueueGuard *guard;
  LevelZeroQueue *queue;
  // Friendship definition needed to access private constructor.
  friend class LevelZeroQueueGuard;
};

/// Class guarding access to LevelZeroQueue that can have only one user,
/// represented as LevelZeroQueueUser.
class LevelZeroQueueGuard {
public:
  explicit LevelZeroQueueGuard(LevelZeroQueue *queue, bool init = false)
      : used(init), queue(queue) {}
  ~LevelZeroQueueGuard() {delete queue;}

  /// Returns properties of contained queue.
  ze_command_queue_group_properties_t getLevelZeroProperties() {
    return queue->getLevelZeroProperties();
  }
  /// Checks if this queue is currently in use. Does not guarantee that
  /// subsequent calls to use() will return non-empty user in
  /// multithreaded scenerios.
  bool isUsed();
  /// Tries to reserve queue for use and returns corresponding user.
  /// If queue is currently in use and cannot be reserved returns
  /// empty user.
  LevelZeroQueueUser use();

private:
  std::atomic<bool> used;
  LevelZeroQueue *queue;
  // Fiendship definition to release guard when upon user destruction.
  friend class LevelZeroQueueUser;
};

/// LevelZero device abstraction.
/// Purpose of this class is to manage LevelZero resources connected
/// to device.
class LevelZeroDevice final
    : public pmlc::rt::Device,
      public std::enable_shared_from_this<LevelZeroDevice> {
public:
  explicit LevelZeroDevice(ze_device_handle_t device);
  ~LevelZeroDevice();

  std::unique_ptr<Executable>
  compile(const std::shared_ptr<pmlc::compiler::Program> &program) final;

  /// Returns LevelZero context created with only this device.
  ze_context_handle_t getLevelZeroContext() { return context; }
  /// Returns LevelZero device.
  ze_device_handle_t getLevelZeroDevice() { return device; }
  /// Returns LevelZero queue with specified properties for execution
  /// on this device. Returned user is always non-empty.
  LevelZeroQueueUser getQueue(ze_command_queue_group_properties_t properties);
  void clearQueues();

private:
  ze_context_handle_t context;
  ze_device_handle_t device;
  std::vector<std::unique_ptr<LevelZeroQueueGuard>> queues;
  std::mutex queuesMutex;
};

} // namespace pmlc::rt::level_zero
