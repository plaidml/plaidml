// Copyright 2020 Intel Corporation
#include "pmlc/rt/opencl/opencl_device.h"

#include "pmlc/rt/jit_executable.h"
#include "pmlc/util/logging.h"

namespace pmlc::rt::opencl {

OpenCLQueue::OpenCLQueue(const cl::Context &context, const cl::Device &device,
                         cl::QueueProperties properties)
    : queue(context, device, properties), properties(properties) {}

OpenCLQueueUser::OpenCLQueueUser() : OpenCLQueueUser(nullptr, nullptr) {}

OpenCLQueueUser::OpenCLQueueUser(OpenCLQueueGuard *guard, OpenCLQueue *queue)
    : guard(guard), queue(queue) {}

OpenCLQueueUser::OpenCLQueueUser(OpenCLQueueUser &&move)
    : guard(move.guard), queue(move.queue) {
  move.guard = nullptr;
  move.queue = nullptr;
}

OpenCLQueueUser::~OpenCLQueueUser() {
  if (guard != nullptr)
    guard->used.store(false, std::memory_order_relaxed);
}

bool OpenCLQueueGuard::isUsed() { return used.load(std::memory_order_relaxed); }

OpenCLQueueUser OpenCLQueueGuard::use() {
  bool expected = false;
  bool desired = true;
  used.compare_exchange_strong(expected, desired, std::memory_order_relaxed,
                               std::memory_order_relaxed);
  // Currently in use - return null user.
  if (expected)
    return OpenCLQueueUser();
  return OpenCLQueueUser(this, &queue);
}

OpenCLDevice::OpenCLDevice(cl::Device device)
    : context(device), device(device) {
  IVLOG(1, "Instantiating OpenCL device: " << device.getInfo<CL_DEVICE_NAME>());
}

std::unique_ptr<Executable>
OpenCLDevice::compile(const std::shared_ptr<pmlc::compiler::Program> &program) {
  return makeJitExecutable(program, shared_from_this(),
                           mlir::ArrayRef<void *>{this});
}

OpenCLQueueUser OpenCLDevice::getQueue(cl::QueueProperties properties) {
  // Lock modification of queues vector.
  std::lock_guard<std::mutex> lock(queuesMutex);
  for (std::unique_ptr<OpenCLQueueGuard> &guard : queues) {
    if (guard->getOclProperties() != properties || guard->isUsed())
      continue;
    OpenCLQueueUser user = guard->use();
    if (user)
      return user;
  }
  // Because queues is locked and not visible to other threads yet it
  // is safe to assume that new guard will return non-empty OpenCLQueueUser.
  OpenCLQueue newQueue(context, device, properties);
  queues.emplace_back(std::make_unique<OpenCLQueueGuard>(newQueue));
  return queues.back()->use();
}

} // namespace pmlc::rt::opencl
