// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/opencl/event.h"

#include <utility>

#include "base/util/error.h"
#include "tile/hal/opencl/result.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

const char* EventCommandTypeStr(cl_command_type code);

std::shared_ptr<Event> Event::Downcast(const std::shared_ptr<hal::Event>& event, const CLObj<cl_context>& cl_ctx) {
  std::shared_ptr<Event> evt = std::dynamic_pointer_cast<Event>(event);
  if (!evt || evt->cl_ctx_ != cl_ctx) {
    LOG(ERROR) << "Incompatible event for Tile device. event: " << evt;
    throw error::InvalidArgument{"Incompatible event for Tile device"};
  }
  return evt;
}

std::vector<cl_event> Event::Downcast(const std::vector<std::shared_ptr<hal::Event>>& events,
                                      const CLObj<cl_context>& cl_ctx, const DeviceState::Queue& queue) {
  std::vector<cl_event> result;
  for (const auto& event : events) {
    auto evt = Downcast(event, cl_ctx);
    if (evt->cl_event_ && (evt->queue_ != &queue || queue.props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)) {
      std::lock_guard<std::mutex> lock{evt->state_->mu};
      if (!evt->state_->completed) {
        result.emplace_back(evt->cl_event_.get());
      }
    }
  }
  return result;
}

boost::future<std::vector<std::shared_ptr<hal::Result>>> Event::WaitFor(
    const std::vector<std::shared_ptr<hal::Event>>& events, const std::shared_ptr<DeviceState>& device_state) {
  std::vector<cl_event> mdeps;
  std::vector<std::shared_ptr<Event>> hal_events;
  for (const auto& event : events) {
    auto evt = Downcast(event, device_state->cl_ctx());
    if (evt->cl_event_) {
      mdeps.emplace_back(evt->cl_event_.get());
      hal_events.emplace_back(std::move(evt));
    }
  }
  if (!mdeps.size()) {
    std::vector<std::shared_ptr<hal::Result>> results;
    return boost::make_ready_future(std::move(results));
  }
  CLObj<cl_event> evt;
  const auto& queue = device_state->cl_normal_queue();
  Err err = clEnqueueMarkerWithWaitList(queue.cl_queue.get(),  // command_queue
                                        mdeps.size(),          // num_events_in_wait_list
                                        mdeps.data(),          // event_wait_list
                                        evt.LvaluePtr());      // event
  Err::Check(err, "Failed to synchronize work queue");
  context::Context ctx{};
  Event event{ctx, device_state, std::move(evt), queue};
  auto future = event.GetFuture();
  auto results = future.then([hal_events = std::move(hal_events), device_state](decltype(future) fut) {
    std::vector<std::shared_ptr<hal::Result>> results;
    results.reserve(hal_events.size());
    try {
      fut.get();
    } catch (...) {
      LOG(ERROR) << boost::current_exception();
    }
    for (const auto& event : hal_events) {
      if (event->state_->result) {
        results.emplace_back(event->state_->result);
      }
    }
    return results;
  });
  device_state->FlushCommandQueue();
  return results;
}

Event::Event(const context::Context& ctx, const std::shared_ptr<DeviceState>& device_state, CLObj<cl_event> cl_event,
             const DeviceState::Queue& queue)
    : Event(ctx, device_state, cl_event, queue, std::make_shared<Result>(ctx, device_state, cl_event)) {}

Event::Event(const context::Context& ctx, const std::shared_ptr<DeviceState>& device_state, CLObj<cl_event> cl_event,
             const DeviceState::Queue& queue, const std::shared_ptr<hal::Result>& result)
    : queue_{&queue},
      cl_ctx_{device_state->cl_ctx()},
      cl_event_{std::move(cl_event)},
      state_{std::make_shared<FutureState>()} {
  state_->result = result;
  if (!cl_event_) {
    state_->prom.set_value(state_->result);
  }
}

Event::~Event() {
  if (cl_event_ && !started_) {
    state_->prom.set_value(std::shared_ptr<hal::Result>());
  }
}

boost::shared_future<std::shared_ptr<hal::Result>> Event::GetFuture() {
  std::lock_guard<std::mutex> lock{mu_};
  if (!cl_event_) {
    return boost::make_ready_future(state_->result);
  }

  if (!started_) {
    {
      // Technically, we don't need to hold this lock while accessing
      // state_->self, since there's no way we can access it unsafely
      // -- but it's nice to be explicit and careful with our
      // synchronization.
      std::lock_guard<std::mutex> lock{state_->mu};
      if (!fut_.valid()) {
        fut_ = state_->prom.get_future().share();
      }
      state_->self = state_;
    }

    try {
      Err err = clSetEventCallback(cl_event_.get(), CL_COMPLETE, &EventComplete, state_.get());
      Err::Check(err, "Unable to register an event callback");
    } catch (...) {
      std::lock_guard<std::mutex> lock{state_->mu};
      state_->self.reset();
      throw;
    }

    started_ = true;
  }

  return fut_;
}

void Event::EventComplete(cl_event evt, cl_int status, void* data) {
  auto state = static_cast<FutureState*>(data);

  std::shared_ptr<FutureState> self_ref;

  {
    std::lock_guard<std::mutex> lock{state->mu};
    state->completed = true;
    self_ref = std::move(state->self);
  }

  try {
    if (status < 0) {
      Err err(status);
      cl_command_type type = 0;
      clGetEventInfo(evt, CL_EVENT_COMMAND_TYPE, sizeof(type), &type, nullptr);
      LOG(ERROR) << "Event " << EventCommandTypeStr(type) << " failed with: " << err.str();
      Err::Check(err, "Event completed with failure");
    }
    state->prom.set_value(state->result);
  } catch (...) {
    state->prom.set_exception(boost::current_exception());
  }

  // N.B. state may be deleted as we leave this context.
}

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
