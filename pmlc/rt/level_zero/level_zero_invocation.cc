// Copyright 2020 Intel Corporation
#include "pmlc/rt/level_zero/level_zero_invocation.h"

#include <algorithm>
#include <chrono>
#include <string>
#include <utility>

#include "pmlc/util/logging.h"

namespace pmlc::rt::level_zero {

void LevelZeroMemory::enqueueRead(ze_command_list_handle_t list, void *dst,
                                  std::vector<ze_event_handle_t> &dependencies,
                                  ze_event_handle_t &resultE) {
  lzu::append_memory_copy(list, dst, buffer, bytes, resultE,
                          dependencies.size(), dependencies.data());
}

void LevelZeroMemory::enqueueWrite(ze_command_list_handle_t list, void *src,
                                   std::vector<ze_event_handle_t> &dependencies,
                                   ze_event_handle_t &resultE) {
  lzu::append_memory_copy(list, buffer, src, bytes, resultE,
                          dependencies.size(), dependencies.data());
}

LevelZeroKernel::LevelZeroKernel(ze_module_handle_t module, std::string name)
    : module(module), name(name) {
  kernel = lzu::create_function(module, 0, name);
}

LevelZeroKernel::~LevelZeroKernel() {
  lzu::destroy_function(kernel);
  lzu::destroy_module(module);
}

void LevelZeroKernel::addDependency(LevelZeroEvent *event) {
  dependencies.push_back(event->getEvent());
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
}

LevelZeroEvent::LevelZeroEvent(ze_event_handle_t event,
                               LevelZeroActionKind kind, std::string name)
    : event(event), kind(kind), name(std::move(name)) {}

void LevelZeroEvent::wait(const std::vector<LevelZeroEvent *> &events) {
  std::vector<ze_event_handle_t> zeEvents;
  std::transform(events.begin(), events.end(), std::back_inserter(zeEvents),
                 [](const LevelZeroEvent *event) { return event->getEvent(); });
  for (auto e : zeEvents) {
    zeEventHostSynchronize(e, UINT64_MAX);
  }
}
ze_command_queue_group_properties_t p;
LevelZeroInvocation::LevelZeroInvocation(LevelZeroDevice *device)
    : device{device->shared_from_this()}, queueUser(device->getQueue(p)) {
  eventPool.InitEventPool(device->getLevelZeroContext(), 100);
  // ze_command_queue_group_properties_t p;
  // p.flags = ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE;
  // queueUser = device->getQueue(p);
}

LevelZeroInvocation::~LevelZeroInvocation() {
  // Need to explicitly wait for all operations to avoid unfinished events
  // when gathering profiling information.
  finish();

  for (size_t i = 0; i < memories.size(); i++) {
    delete memories[i];
  }
  for (std::unique_ptr<LevelZeroEvent> &event : events) {
    eventPool.destroy_event(event->getEvent());
  }
  for (size_t i = 0; i < kernels.size(); i++) {
    delete kernels[i];
  }
  // device->clearQueues();
#if 0
  // Gather profiling information.
  using std::chrono::nanoseconds;
  using fp_milliseconds =
      std::chrono::duration<double, std::chrono::milliseconds::period>;
  // Calculate total time as difference between earliest enqueue
  // and latest execution end.
  cl_ulong allQueued = static_cast<cl_ulong>(-1);
  cl_ulong allEnd = 0;
  nanoseconds totalExecuteTime{0};
  nanoseconds kernelExecuteTime{0};
  nanoseconds memoryExecuteTime{0};
  unsigned kernelsCnt = 0;
  unsigned memoryCnt = 0;
  for (std::unique_ptr<LevelZeroEvent> &event : events) {
    cl::Event oclEvent = event->getEvent();
    cl_ulong queued = oclEvent.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
    cl_ulong start = oclEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end = oclEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    allQueued = std::min(allQueued, queued);
    allEnd = std::max(allEnd, end);

    nanoseconds executeTime{end - start};
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
  nanoseconds totalTime{allEnd - allQueued};
  IVLOG(1, "Total LevelZero time: " << fp_milliseconds(totalTime).count() << "ms");
  IVLOG(1, "Total LevelZero execute time: "
               << fp_milliseconds(totalExecuteTime).count() << "ms");
  IVLOG(1, "Total LevelZero kernels: " << kernelsCnt);
  IVLOG(1, "Total LevelZero kernel execute time: "
               << fp_milliseconds(kernelExecuteTime).count() << "ms");
  IVLOG(1, "Total LevelZero memory transfers: " << memoryCnt);
  IVLOG(1, "Total LevelZero memory transfer time: "
               << fp_milliseconds(memoryExecuteTime).count() << "ms");

  device->execTimeInMS = fp_milliseconds(totalExecuteTime).count();
#endif
}

LevelZeroMemory *LevelZeroInvocation::allocateMemory(size_t bytes) {
  // TODO host memory, device memory
  void *buffer =
      lzu::allocate_shared_memory(bytes, 1, 0, 0, device->getLevelZeroDevice(),
                                  device->getLevelZeroContext());
  return new LevelZeroMemory(buffer, bytes, device->getLevelZeroContext());
}

void LevelZeroInvocation::deallocateMemory(LevelZeroMemory *memory) {
  // delete memory;
  memories.push_back(memory);
}

LevelZeroEvent *
LevelZeroInvocation::enqueueRead(LevelZeroMemory *src, void *dst,
                                 const std::vector<LevelZeroEvent *> &deps) {
  std::vector<ze_event_handle_t> dependencies;
  std::transform(deps.begin(), deps.end(), std::back_inserter(dependencies),
                 [](const LevelZeroEvent *event) { return event->getEvent(); });
  ze_event_handle_t event;
  eventPool.create_event(event);
  src->enqueueRead(queueUser.getLevelZeroList(), dst, dependencies, event);
  return wrapEvent(event, LevelZeroActionKind::Read, "read");
}

LevelZeroEvent *
LevelZeroInvocation::enqueueWrite(LevelZeroMemory *dst, void *src,
                                  const std::vector<LevelZeroEvent *> &deps) {
  std::vector<ze_event_handle_t> dependencies;
  std::transform(deps.begin(), deps.end(), std::back_inserter(dependencies),
                 [](const LevelZeroEvent *event) { return event->getEvent(); });
  ze_event_handle_t event;
  eventPool.create_event(event);
  dst->enqueueWrite(queueUser.getLevelZeroList(), src, dependencies, event);
  return wrapEvent(event, LevelZeroActionKind::Write, "write");
}

LevelZeroKernel *LevelZeroInvocation::createKernelFromIL(char *data,
                                                         size_t bytes,
                                                         const char *name) {
  uint8_t *buf = reinterpret_cast<uint8_t *>(data);
  ze_module_handle_t module = lzu::create_module(
      device->getLevelZeroContext(), device->getLevelZeroDevice(), buf, bytes,
      ZE_MODULE_FORMAT_IL_SPIRV, "", nullptr);

  LevelZeroKernel *kernel = new LevelZeroKernel(module, name);
  kernels.push_back(kernel);
  return kernel;
}

LevelZeroEvent *LevelZeroInvocation::enqueueKernel(LevelZeroKernel *kernel,
                                                   ze_group_count_t gws,
                                                   ze_group_count_t lws) {
  ze_event_handle_t event;
  eventPool.create_event(event);
  kernel->enqueue(queueUser.getLevelZeroList(), gws, lws, event);
  LevelZeroEvent *result =
      wrapEvent(event, LevelZeroActionKind::Kernel, kernel->getName());
  // delete kernel;
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

void LevelZeroInvocation::flush() {
  ze_command_list_handle_t command_list = queueUser.getLevelZeroList();
  ze_command_queue_handle_t command_queue = queueUser.getLevelZeroQueue();
  lzu::close_command_list(command_list);
  lzu::execute_command_lists(command_queue, 1, &command_list, nullptr);
  lzu::synchronize(command_queue, UINT64_MAX);
  lzu::reset_command_list(command_list);
}

void LevelZeroInvocation::finish() {
  // xin can not wait twice
  lzu::synchronize(queueUser.getLevelZeroQueue(), UINT64_MAX);
}

LevelZeroEvent *LevelZeroInvocation::wrapEvent(ze_event_handle_t event,
                                               LevelZeroActionKind kind,
                                               std::string name) {
  events.emplace_back(
      std::make_unique<LevelZeroEvent>(event, kind, std::move(name)));
  return events.back().get();
}

} // namespace pmlc::rt::level_zero
