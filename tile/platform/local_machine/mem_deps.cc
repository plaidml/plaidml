// Copyright 2017-2018 Intel Corporation.

#include "tile/platform/local_machine/mem_deps.h"

#include <utility>

namespace vertexai {
namespace tile {
namespace local_machine {

void MemDeps::GetReadDependencies(std::vector<std::shared_ptr<hal::Event>>* deps) {
  std::lock_guard<std::mutex> lock{mu_};
  if (ep_) {
    std::rethrow_exception(ep_);
  }
  deps->reserve(deps->size() + events_.size());
  for (const auto& evt : events_) {
    deps->emplace_back(evt);
  }
}

void MemDeps::AddReadDependency(std::shared_ptr<hal::Event> event) {
  boost::shared_future<std::shared_ptr<hal::Result>> fut;
  std::list<std::shared_ptr<hal::Event>>::iterator it;
  {
    std::lock_guard<std::mutex> lock{mu_};
    ep_ = std::exception_ptr{};
    fut = event->GetFuture();
    it = events_.emplace(events_.end(), std::move(event));
  }
  fut.then([self = shared_from_this(), it](boost::shared_future<std::shared_ptr<hal::Result>> future) {
    future.get();
    std::lock_guard<std::mutex> lock{self->mu_};
    self->events_.erase(it);
  });
}

void MemDeps::Poison(std::exception_ptr ep) noexcept {
  std::lock_guard<std::mutex> lock{mu_};
  ep_ = ep;
}

}  // namespace local_machine
}  // namespace tile
}  // namespace vertexai
