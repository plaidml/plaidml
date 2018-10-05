// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cpu/event.h"

#include <utility>

#include "base/util/error.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

Event::Event(boost::shared_future<std::shared_ptr<hal::Result>> future) : future_{std::move(future)} {}

std::shared_ptr<Event> Event::Downcast(const std::shared_ptr<hal::Event>& event) {
  std::shared_ptr<Event> evt = std::dynamic_pointer_cast<Event>(event);
  if (!evt) {
    throw error::InvalidArgument{"Incompatible event for Tile device"};
  }
  return evt;
}

boost::future<std::vector<std::shared_ptr<hal::Result>>> Event::WaitFor(
    const std::vector<std::shared_ptr<hal::Event>>& events) {
  std::vector<boost::shared_future<std::shared_ptr<hal::Result>>> futures;
  for (auto& event : events) {
    futures.emplace_back(event->GetFuture());
  }
  auto deps = boost::when_all(futures.begin(), futures.end());
  auto results = deps.then([](decltype(deps) fut) {
    std::vector<std::shared_ptr<hal::Result>> results;
    for (const auto& result : fut.get()) {
      results.emplace_back(result.get());
    }
    return results;
  });
  return results;
}

boost::shared_future<std::shared_ptr<hal::Result>> Event::GetFuture() { return future_; }

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
