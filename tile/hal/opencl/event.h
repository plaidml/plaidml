// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include "tile/base/hal.h"
#include "tile/hal/opencl/buffer.h"
#include "tile/hal/opencl/device_state.h"
#include "tile/hal/opencl/ocl.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

// Implements hal::Event in terms of OpenCL events.
class Event final : public hal::Event {
 public:
  // Casts a hal::Event to an Event, throwing an exception if the supplied hal::Event isn't an
  // OpenCL event, or if it's an event for a different context.
  static std::shared_ptr<Event> Downcast(const std::shared_ptr<hal::Event>& event, const CLObj<cl_context>& cl_ctx);

  // Casts a vector of hal::Event to a vector of cl_event, throwing an exception if the supplied
  // hal::Event objects aren't OpenCL events, or if any are events for a different context. This
  // method does not retain the returned events; callers must hold onto the input vector to keep the
  // events alive.
  static std::vector<cl_event> Downcast(const std::vector<std::shared_ptr<hal::Event>>& events,
                                        const CLObj<cl_context>& cl_ctx, const DeviceState::Queue& queue);

  // Returns a future that waits for all of the supplied events to complete.
  static boost::future<std::vector<std::shared_ptr<hal::Result>>> WaitFor(
      const std::vector<std::shared_ptr<hal::Event>>& events, const std::shared_ptr<DeviceState>& device_state);

  Event(const context::Context& ctx, const std::shared_ptr<DeviceState>& device_state, CLObj<cl_event> cl_event,
        const DeviceState::Queue& queue);

  Event(const context::Context& ctx, const std::shared_ptr<DeviceState>& device_state, CLObj<cl_event> cl_event,
        const DeviceState::Queue& queue, const std::shared_ptr<hal::Result>& result);

  ~Event() final;

  boost::shared_future<std::shared_ptr<hal::Result>> GetFuture() final;

 private:
  struct FutureState {
    std::mutex mu;
    bool completed = false;
    std::shared_ptr<FutureState> self;  // Set iff clSetEventCallback is in flight
    std::shared_ptr<hal::Result> result;
    boost::promise<std::shared_ptr<hal::Result>> prom;
  };

  static void EventComplete(cl_event evt, cl_int status, void* data);

  const DeviceState::Queue* queue_;
  std::mutex mu_;
  bool started_ = false;
  CLObj<cl_context> cl_ctx_;
  CLObj<cl_event> cl_event_;
  std::shared_ptr<FutureState> state_;
  boost::shared_future<std::shared_ptr<hal::Result>> fut_;
};

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
