// Copyright 2021 Intel Corporation
#include "pmlc/rt/level_zero/level_zero_invocation.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <string>
#include <utility>

#include "pmlc/util/logging.h"

#define UNUSED_VARIABLE(x) (void)(x)

namespace pmlc::rt::level_zero {

void LevelZeroMemory::enqueueRead(ze_command_list_handle_t list, void *dst,
                                  std::vector<ze_event_handle_t> &dependencies,
                                  ze_event_handle_t &resultE) {
  if (LevelZeroMemoryKind::Host == kind) {
    for (auto e : dependencies) {
      if (e != nullptr) {
        zeEventHostSynchronize(e, UINT64_MAX);
      }
    }
    std::memcpy(dst, buffer, bytes);
  } else {
    lzu::append_memory_copy(list, dst, buffer, bytes, resultE,
                            dependencies.size(), dependencies.data());
  }
}

void LevelZeroMemory::enqueueWrite(ze_command_list_handle_t list, void *src,
                                   std::vector<ze_event_handle_t> &dependencies,
                                   ze_event_handle_t &resultE) {
  if (LevelZeroMemoryKind::Host == kind) {
    for (auto e : dependencies) {
      if (e != nullptr) {
        zeEventHostSynchronize(e, UINT64_MAX);
      }
    }
    std::memcpy(buffer, src, bytes);
  } else {
    lzu::append_memory_copy(list, buffer, src, bytes, resultE,
                            dependencies.size(), dependencies.data());
  }
}

LevelZeroKernel::LevelZeroKernel(ze_module_handle_t module, std::string name)
    : module(module), name(name) {
  kernel = lzu::create_function(module, /*flag*/ 0, name);
}

LevelZeroKernel::~LevelZeroKernel() {
  lzu::destroy_function(kernel);
  lzu::destroy_module(module);
}

void LevelZeroKernel::addDependency(LevelZeroEvent *event) {
  if (event->getEvent() != nullptr) {
    dependencies.push_back(event->getEvent());
  }
}

void LevelZeroKernel::setArg(unsigned idx, LevelZeroMemory *memory) {
  void *buf = reinterpret_cast<int64_t *>(memory->getBuffer());
  lzu::set_argument_value(kernel, idx, sizeof(buf), &buf);
}

void LevelZeroKernel::enqueue(ze_command_list_handle_t list,
                              ze_group_count_t gws, ze_group_count_t lws,
                              ze_event_handle_t &resultE) {
  lzu::append_launch_function(list, kernel, &gws, resultE, dependencies.size(),
                              dependencies.data());
  dependencies.clear();
  UNUSED_VARIABLE(lws);
}

LevelZeroEvent::LevelZeroEvent(LevelZeroInvocation *invocation,
                               ze_event_handle_t event,
                               LevelZeroActionKind kind, std::string name)
    : invocation(invocation), event(event), kind(kind), name(std::move(name)) {
  memset(&timestamp, 0, sizeof(timestamp));
}

void LevelZeroEvent::wait(const std::vector<LevelZeroEvent *> &events) {
  for (auto event : events) {
    ze_event_handle_t e = event->getEvent();
    if (e != nullptr) {
      zeEventHostSynchronize(e, UINT64_MAX);
      // Query timestamp and release event
      ze_kernel_timestamp_result_t value = {};
      zeEventQueryKernelTimestamp(e, &value);
      event->setTimestamp(value);
      event->getInvocation()->releaseZeEvent(e);
      event->setEvent(nullptr);
    }
  }
}

LevelZeroInvocation::LevelZeroInvocation(
    LevelZeroDevice *device, ze_command_queue_group_properties_t prop)
    : device{device->shared_from_this()}, queueUser(device->getQueue(prop)) {
  // Create an eventPool with fixed count, need to release event as soon as
  // possible
  eventPool.InitEventPool(device->getLevelZeroContext(), /*count*/ 600);
}

LevelZeroInvocation::~LevelZeroInvocation() {
  // Need to explicitly wait for all operations to avoid unfinished events
  // when gathering profiling information.
  finish();
  // Release resources.
  for (size_t i = 0; i < memories.size(); i++) {
    delete memories[i];
  }
  memories.clear();
}

LevelZeroMemory *LevelZeroInvocation::allocateMemory(size_t bytes,
                                                     LevelZeroMemoryKind kind) {
  // Can add host memory, device memory based on device config
  void *buffer = nullptr;
  if (LevelZeroMemoryKind::Host == kind) {
    buffer = lzu::allocate_host_memory(bytes, /*alignment*/ 1,
                                       device->getLevelZeroContext());
  } else if (LevelZeroMemoryKind::Device == kind) {
    buffer = lzu::allocate_device_memory(
        bytes, /*alignment*/ 1, /*dev_flags*/ 0, /*ordinal*/ 0,
        device->getLevelZeroDevice(), device->getLevelZeroContext());
  } else {
    buffer = lzu::allocate_shared_memory(
        bytes, /*alignment*/ 1, /*dev_flags*/ 0, /*host_flags*/ 0,
        device->getLevelZeroDevice(), device->getLevelZeroContext());
  }
  return new LevelZeroMemory(buffer, bytes, kind,
                             device->getLevelZeroContext());
}

void LevelZeroInvocation::deallocateMemory(LevelZeroMemory *memory) {
  // Memory release shall run after kernel excution, postpone to destructor.
  // delete memory;
  memories.push_back(memory);
}

LevelZeroEvent *
LevelZeroInvocation::enqueueRead(LevelZeroMemory *src, void *dst,
                                 const std::vector<LevelZeroEvent *> &deps) {
  std::vector<ze_event_handle_t> dependencies;
  std::transform(deps.begin(), deps.end(), std::back_inserter(dependencies),
                 [](const LevelZeroEvent *event) { return event->getEvent(); });
  ze_event_handle_t event = nullptr;
  if (src->getKind() == LevelZeroMemoryKind::Device ||
      src->getKind() == LevelZeroMemoryKind::Shared) {
    eventPool.create_event(event);
    src->enqueueRead(queueUser.getLevelZeroList(), dst, dependencies, event);
  }
  src->enqueueRead(queueUser.getLevelZeroList(), dst, dependencies, event);
  return wrapEvent(event, LevelZeroActionKind::Read, "read");
}

LevelZeroEvent *
LevelZeroInvocation::enqueueWrite(LevelZeroMemory *dst, void *src,
                                  const std::vector<LevelZeroEvent *> &deps) {
  std::vector<ze_event_handle_t> dependencies;
  std::transform(deps.begin(), deps.end(), std::back_inserter(dependencies),
                 [](const LevelZeroEvent *event) { return event->getEvent(); });
  ze_event_handle_t event = nullptr;
  if (dst->getKind() == LevelZeroMemoryKind::Device ||
      dst->getKind() == LevelZeroMemoryKind::Shared) {
    eventPool.create_event(event);
  }
  dst->enqueueWrite(queueUser.getLevelZeroList(), src, dependencies, event);
  return wrapEvent(event, LevelZeroActionKind::Write, "write");
}

LevelZeroKernel *LevelZeroInvocation::createKernelFromIL(char *data,
                                                         size_t bytes,
                                                         const char *name) {
  uint8_t *buf = reinterpret_cast<uint8_t *>(data);
  ze_module_handle_t module = lzu::create_module(
      device->getLevelZeroContext(), device->getLevelZeroDevice(), buf, bytes,
      ZE_MODULE_FORMAT_IL_SPIRV, /*build_flags*/ "", /*p_build_log*/ nullptr);

  return new LevelZeroKernel(module, name);
}

LevelZeroEvent *LevelZeroInvocation::enqueueKernel(LevelZeroKernel *kernel,
                                                   ze_group_count_t gws,
                                                   ze_group_count_t lws) {
  // In LevelZero, gws is group count, lws is group size.
  lzu::set_group_size(kernel->getKernel(), lws.groupCountX, lws.groupCountY,
                      lws.groupCountZ);
  ze_event_handle_t event;
  eventPool.create_event(event);
  kernel->enqueue(queueUser.getLevelZeroList(), gws, lws, event);
  LevelZeroEvent *result =
      wrapEvent(event, LevelZeroActionKind::Kernel, kernel->getName());
  return result;
}

LevelZeroEvent *
LevelZeroInvocation::enqueueBarrier(const std::vector<LevelZeroEvent *> &deps) {
  ze_event_handle_t result;
  eventPool.create_event(result);
  std::vector<ze_event_handle_t> dependencies;
  std::transform(deps.begin(), deps.end(), std::back_inserter(dependencies),
                 [](const LevelZeroEvent *event) { return event->getEvent(); });
  lzu::append_barrier(queueUser.getLevelZeroList(), result, dependencies.size(),
                      dependencies.data());

  return wrapEvent(result, LevelZeroActionKind::Barrier, "barrier");
}

void LevelZeroInvocation::releaseZeEvent(ze_event_handle_t event) {
  if (event != nullptr) {
    eventPool.destroy_event(event);
  }
}

void LevelZeroInvocation::flush() {
  ze_command_list_handle_t command_list = queueUser.getLevelZeroList();
  ze_command_queue_handle_t command_queue = queueUser.getLevelZeroQueue();
  lzu::close_command_list(command_list);
  lzu::execute_command_lists(command_queue, /*numCommandLists*/ 1,
                             &command_list, /*hfence*/ nullptr);
  lzu::synchronize(command_queue, /*timeout*/ UINT64_MAX);
  lzu::reset_command_list(command_list);
}

void LevelZeroInvocation::finish() {
  lzu::synchronize(queueUser.getLevelZeroQueue(), /*timeout*/ UINT64_MAX);
  // Gather profiling information
  if (events.size() == 0) {
    // All events have been cleared by the latest call of this function.
    return;
  }
  using std::chrono::nanoseconds;
  using fp_milliseconds =
      std::chrono::duration<double, std::chrono::milliseconds::period>;

  // Calculate total time as difference between earliest event
  // and latest event.
  uint64_t allStart = 0;
  uint64_t allEnd = 0;
  nanoseconds totalExecuteTime{0};
  nanoseconds kernelExecuteTime{0};
  nanoseconds memoryExecuteTime{0};
  unsigned kernelsCnt = 0;
  unsigned memoryCnt = 0;

  auto getEventKernelTimestamp =
      [](ze_event_handle_t event) -> ze_kernel_timestamp_result_t {
    ze_kernel_timestamp_result_t value = {};
    zeEventQueryKernelTimestamp(event, &value);
    return value;
  };

  auto deviceProperties = pmlc::rt::level_zero::lzu::get_device_properties(
      device->getLevelZeroDevice());
  const uint64_t timestampFreq = deviceProperties.timerResolution;
  const uint64_t timestampMaxValue =
      ~(-1 << deviceProperties.kernelTimestampValidBits);

  for (std::unique_ptr<LevelZeroEvent> &event : events) {
    ze_kernel_timestamp_result_t timestamp = event->getTimestamp();
    if (event->getEvent() != nullptr) {
      timestamp = getEventKernelTimestamp(event->getEvent());
    }
    uint64_t start = timestamp.context.kernelStart;
    uint64_t end = timestamp.context.kernelEnd;

    allStart = std::min(allStart, start);
    allEnd = std::max(allEnd, end);

    auto eventExecuteTime =
        (end >= start)
            ? (end - start) * timestampFreq
            : ((timestampMaxValue - start + end + 1) * timestampFreq);
    nanoseconds executeTime{eventExecuteTime};
    totalExecuteTime += executeTime;

    if (event->getKind() == LevelZeroActionKind::Kernel) {
      kernelExecuteTime += executeTime;
      kernelsCnt += 1;
      IVLOG(2, "  Kernel '" << event->getName() << "' execute time: "
                            << fp_milliseconds(executeTime).count() << "ms");
    } else if (event->getKind() == LevelZeroActionKind::Read ||
               event->getKind() == LevelZeroActionKind::Write) {
      memoryExecuteTime += executeTime;
      memoryCnt += 1;
      IVLOG(2, "  Memory " << event->getName() << " execute time: "
                           << fp_milliseconds(executeTime).count() << "ms");
    }
  }

  auto allEventTime =
      (allEnd >= allStart)
          ? (allEnd - allStart) * timestampFreq
          : ((timestampMaxValue - allStart + allEnd + 1) * timestampFreq);
  nanoseconds totalTime{allEventTime};
  IVLOG(1, "Total Level Zero time: " << fp_milliseconds(totalTime).count()
                                     << "ms");
  IVLOG(1, "Total Level Zero execution time: "
               << (fp_milliseconds(totalExecuteTime).count()) << "ms");
  IVLOG(1, "Total Level Zero Kernels: " << kernelsCnt);
  IVLOG(1, "Total Level Zero Kernel execution time: "
               << (fp_milliseconds(kernelExecuteTime).count()) << "ms");
  IVLOG(1, "Total Level Zero memory transfers: " << memoryCnt);
  IVLOG(1, "Total Level Zero memory transfer time: "
               << (fp_milliseconds(memoryExecuteTime).count()) << "ms");

  device->execTimeInMS = fp_milliseconds(totalExecuteTime).count();

  for (std::unique_ptr<LevelZeroEvent> &event : events) {
    if (event->getEvent() != nullptr)
      eventPool.destroy_event(event->getEvent());
  }
  events.clear();
}

LevelZeroEvent *LevelZeroInvocation::wrapEvent(ze_event_handle_t event,
                                               LevelZeroActionKind kind,
                                               std::string name) {
  events.emplace_back(
      std::make_unique<LevelZeroEvent>(this, event, kind, std::move(name)));
  return events.back().get();
}

} // namespace pmlc::rt::level_zero
