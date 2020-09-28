// Copyright 2020 Intel Corporation
#include "pmlc/rt/opencl/opencl_invocation.h"

#include <algorithm>
#include <chrono>
#include <string>
#include <utility>

#include "pmlc/util/logging.h"

namespace pmlc::rt::opencl {

cl::Event
OpenCLMemory::enqueueRead(cl::CommandQueue queue, void *dst,
                          const std::vector<cl::Event> &dependencies) {
  cl::Event result;
  queue.enqueueReadBuffer(buffer, /*blocking=*/false, /*offset=*/0, bytes, dst,
                          &dependencies, &result);
  return result;
}

cl::Event
OpenCLMemory::enqueueWrite(cl::CommandQueue queue, void *src,
                           const std::vector<cl::Event> &dependencies) {
  cl::Event result;
  queue.enqueueWriteBuffer(buffer, /*blocking=*/false, /*offset=*/0, bytes, src,
                           &dependencies, &result);
  return result;
}

OpenCLKernel::OpenCLKernel(cl::Program program, std::string name)
    : kernel(program, name.c_str()), name(std::move(name)) {}

void OpenCLKernel::addDependency(OpenCLEvent *event) {
  dependencies.push_back(event->getEvent());
}

void OpenCLKernel::setArg(unsigned idx, OpenCLMemory *memory) {
  kernel.setArg(idx, memory->getBuffer());
}

cl::Event OpenCLKernel::enqueue(cl::CommandQueue queue, cl::NDRange gws,
                                cl::NDRange lws) {
  cl::Event result;
  queue.enqueueNDRangeKernel(kernel, /*offset=*/cl::NDRange(), gws, lws,
                             &dependencies, &result);
  return result;
}

OpenCLEvent::OpenCLEvent(cl::Event event, OpenCLActionKind kind,
                         std::string name)
    : event(event), kind(kind), name(std::move(name)) {}

void OpenCLEvent::wait(const std::vector<OpenCLEvent *> &events) {
  std::vector<cl::Event> oclEvents;
  std::transform(events.begin(), events.end(), std::back_inserter(oclEvents),
                 [](const OpenCLEvent *event) { return event->getEvent(); });
  cl::Event::waitForEvents(oclEvents);
}

OpenCLInvocation::OpenCLInvocation()
    : device{Device::current<OpenCLDevice>()},
      queueUser(device->getQueue(cl::QueueProperties::Profiling)) {}

OpenCLInvocation::~OpenCLInvocation() {
  // Need to explicitly wait for all operations to avoid unfinished events
  // when gathering profiling information.
  finish();
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
  for (std::unique_ptr<OpenCLEvent> &event : events) {
    cl::Event oclEvent = event->getEvent();
    cl_ulong queued = oclEvent.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
    cl_ulong start = oclEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    cl_ulong end = oclEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    allQueued = std::min(allQueued, queued);
    allEnd = std::max(allEnd, end);

    nanoseconds executeTime{end - start};
    totalExecuteTime += executeTime;

    if (event->getKind() == OpenCLActionKind::Kernel) {
      kernelExecuteTime += executeTime;
      kernelsCnt += 1;
      IVLOG(2, "  Kernel '" << event->getName() << "' execute time: "
                            << fp_milliseconds(executeTime).count() << "ms");
    } else if (event->getKind() == OpenCLActionKind::Read ||
               event->getKind() == OpenCLActionKind::Write) {
      memoryExecuteTime += executeTime;
      memoryCnt += 1;
      IVLOG(2, "  Memory " << event->getName() << " execute time: "
                           << fp_milliseconds(executeTime).count() << "ms");
    }
  }
  nanoseconds totalTime{allEnd - allQueued};
  IVLOG(1, "Total OpenCL time: " << fp_milliseconds(totalTime).count() << "ms");
  IVLOG(1, "Total OpenCL execute time: "
               << fp_milliseconds(totalExecuteTime).count() << "ms");
  IVLOG(1, "Total OpenCL kernels: " << kernelsCnt);
  IVLOG(1, "Total OpenCL kernel execute time: "
               << fp_milliseconds(kernelExecuteTime).count() << "ms");
  IVLOG(1, "Total OpenCL memory transfers: " << memoryCnt);
  IVLOG(1, "Total OpenCL memory transfer execute time: "
               << fp_milliseconds(memoryExecuteTime).count() << "ms");
}

OpenCLMemory *OpenCLInvocation::allocateMemory(size_t bytes, void *data) {
  cl_mem_flags flags = CL_MEM_READ_WRITE;
  if (data)
    flags |= CL_MEM_COPY_HOST_PTR;

  cl::Buffer buffer(device->getOclContext(), flags, bytes, data);
  return new OpenCLMemory(buffer, bytes);
}

void OpenCLInvocation::deallocateMemory(OpenCLMemory *memory) { delete memory; }

OpenCLEvent *
OpenCLInvocation::enqueueRead(OpenCLMemory *src, void *dst,
                              const std::vector<OpenCLEvent *> &deps) {
  std::vector<cl::Event> dependencies;
  std::transform(deps.begin(), deps.end(), std::back_inserter(dependencies),
                 [](const OpenCLEvent *event) { return event->getEvent(); });
  cl::Event event =
      src->enqueueRead(queueUser.getOclQueue(), dst, dependencies);
  return wrapEvent(event, OpenCLActionKind::Read, "read");
}

OpenCLEvent *
OpenCLInvocation::enqueueWrite(OpenCLMemory *dst, void *src,
                               const std::vector<OpenCLEvent *> &deps) {
  std::vector<cl::Event> dependencies;
  std::transform(deps.begin(), deps.end(), std::back_inserter(dependencies),
                 [](const OpenCLEvent *event) { return event->getEvent(); });
  cl::Event event =
      dst->enqueueWrite(queueUser.getOclQueue(), src, dependencies);
  return wrapEvent(event, OpenCLActionKind::Write, "write");
}

OpenCLKernel *OpenCLInvocation::createKernelFromIL(char *data, size_t bytes,
                                                   const char *name) {
  std::vector<char> source(data, data + bytes);
  cl::Program program(device->getOclContext(), source, /*build=*/true);
  return new OpenCLKernel(program, name);
}

OpenCLEvent *OpenCLInvocation::enqueueKernel(OpenCLKernel *kernel,
                                             cl::NDRange gws, cl::NDRange lws) {
  cl::Event event = kernel->enqueue(queueUser.getOclQueue(), gws, lws);
  OpenCLEvent *result =
      wrapEvent(event, OpenCLActionKind::Kernel, kernel->getName());
  delete kernel;
  return result;
}

OpenCLEvent *
OpenCLInvocation::enqueueBarrier(const std::vector<OpenCLEvent *> &deps) {
  cl::Event result;
  std::vector<cl::Event> dependencies;
  std::transform(deps.begin(), deps.end(), std::back_inserter(dependencies),
                 [](const OpenCLEvent *event) { return event->getEvent(); });
  queueUser.getOclQueue().enqueueBarrierWithWaitList(&dependencies, &result);
  return wrapEvent(result, OpenCLActionKind::Barrier, "barrier");
}

void OpenCLInvocation::flush() { queueUser.getOclQueue().flush(); }

void OpenCLInvocation::finish() { queueUser.getOclQueue().finish(); }

OpenCLEvent *OpenCLInvocation::wrapEvent(cl::Event event, OpenCLActionKind kind,
                                         std::string name) {
  events.emplace_back(
      std::make_unique<OpenCLEvent>(event, kind, std::move(name)));
  return events.back().get();
}

} // namespace pmlc::rt::opencl
