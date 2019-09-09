// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/event.h"

#include <utility>

#include "base/util/error.h"
#include "tile/hal/cm/result.h"
#include "tile/hal/cm/runtime.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

std::shared_ptr<Event> Event::Downcast(const std::shared_ptr<hal::Event>& event) {
  std::shared_ptr<Event> evt = std::dynamic_pointer_cast<Event>(event);
  return evt;
}

std::vector<CmEvent*> Event::Downcast(const std::vector<std::shared_ptr<hal::Event>>& events, const CmQueue* queue) {
  std::vector<CmEvent*> result;
  for (const auto& event : events) {
    auto evt = Downcast(event);
    if (evt->cm_event_ && (evt->queue_ != queue)) {
      std::lock_guard<std::mutex> lock{evt->state_->mu};
      if (!evt->state_->completed) {
        result.emplace_back(evt->cm_event_);
      }
    }
  }
  return result;
}

boost::future<std::vector<std::shared_ptr<hal::Result>>> Event::WaitFor(
    const std::vector<std::shared_ptr<hal::Event>>& events, std::shared_ptr<DeviceState> device_state) {
  std::vector<CmEvent*> mdeps;
  std::vector<std::shared_ptr<Event>> hal_events;
  for (const auto& event : events) {
    std::shared_ptr<Event> evt = Downcast(event);
    if (evt->cm_event_) {
      mdeps.emplace_back(evt->cm_event_);
      hal_events.emplace_back(std::move(evt));
    }
  }
  if (!mdeps.size()) {
    std::vector<std::shared_ptr<hal::Result>> results;
    return boost::make_ready_future(std::move(results));
  }
  CmEvent* e = nullptr;
  const auto& queue = device_state->cmqueue();
  context::Context ctx{};
  Event event{ctx, device_state, e, queue};
  auto future = event.GetFuture();
  auto results = future.then([ hal_events = std::move(hal_events), device_state ](decltype(future) fut) {
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

Event::Event(const context::Context& ctx, std::shared_ptr<DeviceState> device_state, CmEvent* cm_event,
             const CmQueue* queue)
    : Event(ctx, device_state, cm_event, queue, std::make_shared<Result>(ctx, device_state, cm_event)) {}

Event::Event(const context::Context& ctx, std::shared_ptr<DeviceState> device_state, CmEvent* cm_event,
             const CmQueue* queue, const std::shared_ptr<hal::Result>& result)
    : ctx_{ctx}, queue_{queue}, cm_event_{cm_event}, state_{std::make_shared<FutureState>()} {
  state_->result = result;
  if (!cm_event_) {
    state_->prom.set_value(state_->result);
  }
}

Event::~Event() {
  if (cm_event_ && !started_) {
    state_->prom.set_value(std::shared_ptr<hal::Result>());
  }
}

boost::shared_future<std::shared_ptr<hal::Result>> Event::GetFuture() {
  std::lock_guard<std::mutex> lock{mu_};
  if (!cm_event_) {
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
    EventComplete(cm_event_, 0, reinterpret_cast<void*>(state_.get()));
    started_ = true;
  }
  return fut_;
}

void Event::EventComplete(CmEvent* evt, int status, void* data) {
  auto state = static_cast<FutureState*>(data);

  std::shared_ptr<FutureState> self_ref;

  {
    std::lock_guard<std::mutex> lock{state->mu};
    state->completed = true;
    self_ref = std::move(state->self);
  }

  try {
    state->prom.set_value(state->result);
  } catch (...) {
    state->prom.set_exception(boost::current_exception());
  }
  // N.B. state may be deleted as we leave this context.
}

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
