// Copyright 2018 Intel Corporation.

#include "base/context/gate.h"

#include "base/util/error.h"
#include "base/util/logging.h"

namespace vertexai {
namespace context {

Rundown::~Rundown() noexcept {
  if (gate_) {
    gate_->RemoveRundown(handle_);
  }
}

void Rundown::TryEnterGate(std::shared_ptr<Gate> gate) {
  if (gate_) {
    throw error::Internal{"Using a single rundown to track multiple synchronization gates"};
  }
  handle_ = gate->TryAddRundown(std::move(callback_));
  gate_ = std::move(gate);
}

bool Gate::is_open() noexcept {
  std::lock_guard<std::mutex> lock{mu_};
  return is_open_;
}

void Gate::CheckIsOpen() {
  if (!is_open()) {
    throw error::Cancelled{};
  }
}

boost::shared_future<void> Gate::Close() noexcept {
  std::list<std::unique_ptr<Rundown::Callback>> rundowns;
  boost::shared_future<void> result;
  {
    std::lock_guard<std::mutex> lock{mu_};
    if (!is_open_) {
      return finalized_future_;
    }
    is_open_ = false;
    rundowns.swap(rundowns_);
    rundowns_remaining_ = rundowns.size();
    result = finalized_future_;
  }  // Drop the mutex
  for (auto& r : rundowns) {
    if (r) {
      r->OnClose();
    }
  }
  bool done = false;
  {
    std::lock_guard<std::mutex> lock{mu_};
    done = (rundowns_remaining_ == 0);
    close_complete_ = true;
  }
  cv_.notify_all();
  if (done) {
    finalized_prom_.set_value();
  }
  return result;
}

std::list<std::unique_ptr<Rundown::Callback>>::iterator Gate::TryAddRundown(
    std::unique_ptr<Rundown::Callback> callback) {
  std::lock_guard<std::mutex> lock{mu_};
  if (!is_open_) {
    throw error::Cancelled{};
  }
  return rundowns_.emplace(rundowns_.end(), std::move(callback));
}

void Gate::RemoveRundown(std::list<std::unique_ptr<Rundown::Callback>>::iterator handle) noexcept {
  bool done = false;
  {
    std::unique_lock<std::mutex> lock{mu_};
    if (is_open_) {
      rundowns_.erase(handle);
      return;
    }
    // N.B. We need to wait for the close to actually complete, because until it does,
    // the callback may be scheduled to run; removing the rundown while !close_complete_
    // might tear down the datastructures needed by the callback.
    //
    // (As an alternative design, we could require the callback object to maintain
    // references to the objects it might access.  It turns out this complicates the
    // callback object implementation; you wind up with a refcounted subobject, and
    // when writing the callback, you need to consider the possibility that spurious
    // callbacks may occur well after the rundown has been removed.)
    cv_.wait(lock, [this]() { return close_complete_; });
    if (!rundowns_remaining_) {
      LOG(FATAL) << "Over-dereference of a synchronization gate";
    }
    if (!--rundowns_remaining_) {
      done = true;
    }
  }  // Drop the mutex
  if (done) {
    finalized_prom_.set_value();
  }
}

}  // namespace context
}  // namespace vertexai
