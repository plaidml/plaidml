// Copyright 2017, Vertex.AI. CONFIDENTIAL

#include "tile/hal/opencl/event.h"

#include <utility>

#include "base/util/error.h"
#include "tile/hal/opencl/result.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

std::shared_ptr<Event> Event::Upcast(const std::shared_ptr<hal::Event>& event, const CLObj<cl_context>& cl_ctx) {
  std::shared_ptr<Event> evt = std::dynamic_pointer_cast<Event>(event);
  if (!evt || evt->cl_ctx_ != cl_ctx) {
    throw error::InvalidArgument{"Incompatible event for Tile device"};
  }
  return evt;
}

std::vector<cl_event> Event::Upcast(const std::vector<std::shared_ptr<hal::Event>>& events,
                                    const CLObj<cl_context>& cl_ctx, const DeviceState::Queue& queue) {
  std::vector<cl_event> result;
  for (const auto& event : events) {
    auto evt = Upcast(event, cl_ctx);
    if (evt->cl_event_ && (evt->queue_ != &queue || queue.props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)) {
      std::lock_guard<std::mutex> lock{evt->state_->mu};
      if (!evt->state_->completed) {
        result.emplace_back(evt->cl_event_.get());
      }
    }
  }
  return result;
}

boost::future<void> Event::WaitFor(const std::vector<std::shared_ptr<hal::Event>>& events,
                                   const std::shared_ptr<DeviceState>& device_state) {
  if (!events.size()) {
    return boost::make_ready_future();
  }
  if (events.size() == 1) {
    return Upcast(events[0], device_state->cl_ctx())
        ->GetFuture()
        .then([](boost::shared_future<std::shared_ptr<hal::Result>> f) { f.get(); });
  }
  std::vector<cl_event> mdeps;
  std::shared_ptr<Event> last_event;
  for (const auto& event : events) {
    auto evt = Upcast(event, device_state->cl_ctx());
    if (evt->cl_event_) {
      mdeps.emplace_back(evt->cl_event_.get());
      last_event = std::move(evt);
    }
  }
  if (!mdeps.size()) {
    return boost::make_ready_future();
  }
  if (mdeps.size() == 1) {
    return last_event->GetFuture().then([](boost::shared_future<std::shared_ptr<hal::Result>> f) { f.get(); });
  }
  CLObj<cl_event> evt;
  Err err = clEnqueueMarkerWithWaitList(device_state->cl_normal_queue().cl_queue.get(),  // command_queue
                                        mdeps.size(),                                    // num_events_in_wait_list
                                        mdeps.data(),                                    // event_wait_list
                                        evt.LvaluePtr());                                // event
  Err::Check(err, "Failed to synchronize work queue");
  auto result =
      Event{context::Context{}, device_state, std::move(evt), device_state->cl_normal_queue()}.GetFuture().then(
          [](boost::shared_future<std::shared_ptr<hal::Result>> f) { f.get(); });
  device_state->cl_normal_queue().Flush();
  return result;
}

Event::Event(const context::Context& ctx, const std::shared_ptr<DeviceState>& device_state, CLObj<cl_event> cl_event,
             const DeviceState::Queue& queue)
    : queue_{&queue},
      cl_ctx_{device_state->cl_ctx()},
      cl_event_{cl_event},
      state_{std::make_shared<FutureState>()},
      fut_{state_->prom.get_future().share()} {
  state_->result = std::make_shared<Result>(ctx, device_state, std::move(cl_event));
}

boost::shared_future<std::shared_ptr<hal::Result>> Event::GetFuture() {
  std::lock_guard<std::mutex> lock{mu_};
  if (!cl_event_) {
    return boost::make_ready_future<std::shared_ptr<hal::Result>>(state_->result);
  }

  if (!started_) {
    {
      // Technically, we don't need to hold this lock while accessing
      // state_->self, since there's no way we can access it unsafely
      // -- but it's nice to be explicit and careful with our
      // synchronization.
      std::lock_guard<std::mutex> lock{state_->mu};
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

  state->prom.set_value(std::move(state->result));

  // N.B. state may be deleted as we leave this context.
}

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
