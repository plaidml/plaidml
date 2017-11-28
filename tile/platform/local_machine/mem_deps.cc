// Copyright 2017, Vertex.AI.

#include "tile/platform/local_machine/mem_deps.h"

#include <utility>

namespace vertexai {
namespace tile {
namespace local_machine {

std::vector<std::shared_ptr<hal::Event>> MemDeps::GetReadDependencies() {
  std::lock_guard<std::mutex> lock{mu_};
  if (ep_) {
    std::rethrow_exception(ep_);
  }
  std::vector<std::shared_ptr<hal::Event>> result;
  result.reserve(events_.size());
  for (const auto& evt : events_) {
    result.emplace_back(evt);
  }
  return result;
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
  fut.then([ self = shared_from_this(), it ](boost::shared_future<std::shared_ptr<hal::Result>> future) {
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
