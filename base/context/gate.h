// Copyright 2018 Intel Corporation.

#pragma once

#include <condition_variable>
#include <list>
#include <memory>
#include <mutex>
#include <utility>

#include <boost/thread/future.hpp>

#include "base/util/compat.h"

namespace vertexai {
namespace context {

// Gate keeps track of whether a set of activities should continue making progress (one might think of the activities as
// being "gated").
//
// Gates implement a simple finite state machine:
//
//   * -> Open -> Closed -> Finalized -> []
//
//   Open: The initial state.  In this state, new operations may enter the gate.
//
//   Closed: In this state, new operations may not enter the gate, but there may be existing operations still running.
//
//   Finalized: The terminal state.  In this state, no operations are using the gate.
//
// There are a few typical ways to use gates:
//
//   * Long-running operations can periodically invoke CheckIsOpen to throw an exception if the gate's been closed.
//
//   * Operations can prevent the Gate from reaching the Finalized state by allocating a Rundown object and calling
//   TryEnterGate() (which will throw if the gate is closed).  Additionally, Rundown instances can be given a callback
//   to be invoked when the Gate transitions from Open to Closed, allowing operations to actively initiate their own
//   cancellation (cancelling RPCs, closing downstream gates, &c).
class Gate;

class Rundown final {
 public:
  // Constructs a Rundown without an associated callback.
  Rundown() noexcept {}

  // Destroys the Rundown:
  //   If the Rundown's been added to a Gate:
  //     Remove the Rundown's callback (if any) from the Gate (potentially blocking while the callback is evaluated)
  //     Release the gate (allowing it to reach the Finalized state)
  //   Destroy the callback.
  ~Rundown() noexcept;

  // Constructs a Rundown with an associated callback, to be invoked if/when the gate is closed.
  // T must implement the Callable and MoveAssignable concepts, and it must not throw when invoked.
  template <typename CB>
  explicit Rundown(CB callback) : Rundown{} {
    callback_ = compat::make_unique<TypedCallback<CB>>(std::move(callback));
  }

  Rundown(Rundown&& other) noexcept = default;             // MoveConstructible
  Rundown& operator=(Rundown&& other) noexcept = default;  // MoveAssignable

  // Attempt to add the Rundown to the indicated Gate, throwing error::Cancelled if the gate's been closed
  // and error::Internal if the Rundown's already associated with a Gate.
  void TryEnterGate(std::shared_ptr<Gate> gate);

 private:
  friend class Gate;

  struct Callback {
    virtual ~Callback() {}
    virtual void OnClose() noexcept = 0;
  };

  template <typename CB>
  class TypedCallback final : public Callback {
   public:
    explicit TypedCallback(CB cb) : cb_{std::move(cb)} {}

    void OnClose() noexcept final { cb_(); }

   private:
    CB cb_;
  };

  void OnClose() noexcept {
    if (callback_) {
      callback_->OnClose();
    }
  }

  // The callback to invoke when the Gate is closed.  When there is an
  // associated Gate, this is moved into the gate's rundown list.
  std::unique_ptr<Callback> callback_;

  // The Rundown's associated Gate.  Rundowns have at most one gate in their lifetime; once set, the gate is only
  // released when the Rundown is destroyed.
  std::shared_ptr<Gate> gate_;

  // The Rundown's callback in the Gate's rundown list; valid iff gate_ is valid.
  std::list<std::unique_ptr<Callback>>::iterator handle_;
};

class Gate final {
 public:
  // Polls to check whether the gate is currently open.
  bool is_open() noexcept;

  // Polls to see whether the gate has been closed, throwing error::Cancelled if it's been closed.
  void CheckIsOpen();

  // Closes the gate.  Downstream contexts will view the gate as being cancelled, and attempts to enter the gate will
  // fail.  The returned future will be resolved when all asynchronous activity has been completed and has left the
  // gate.
  //
  // It is legal to call this multiple times for a single gate; calling it on a closed gate is a no-op.
  boost::shared_future<void> Close() noexcept;

 private:
  friend class Rundown;

  // Attempt to add a rundown to the gate.  The callback may be empty; if it is not, the callback will be invoked if the
  // gate is closed before the rundown is destroyed (technically, before RemoveRundown returns).  If the gate has been
  // closed, a Cancelled exception will be thrown; otherwise, the iterator is the Gate's reference to the rundown's
  // callback.
  std::list<std::unique_ptr<Rundown::Callback>>::iterator TryAddRundown(std::unique_ptr<Rundown::Callback> callback);

  // Remove the rundown from the gate.
  // Note that this races with closing the gate; if the gate is closing, this will block while the associated callback
  // is invoked.
  void RemoveRundown(std::list<std::unique_ptr<Rundown::Callback>>::iterator handle) noexcept;

  std::mutex mu_;
  std::condition_variable cv_;
  bool is_open_ = true;
  std::list<std::unique_ptr<Rundown::Callback>> rundowns_;
  std::size_t rundowns_remaining_ = 0;
  bool close_complete_ = false;
  boost::promise<void> finalized_prom_;
  boost::shared_future<void> finalized_future_ = finalized_prom_.get_future();
};

}  // namespace context
}  // namespace vertexai
