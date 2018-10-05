// Copyright 2018 Intel Corporation.

#pragma once

#include <cassert>
#include <condition_variable>
#include <mutex>

namespace vertexai {

// Sync automates the implementation of C APIs that operate in synchronous and
// asynchronous modes.  Typically, if no callback method is passed in, the API
// allocates a Sync object on the stack, invokes itself with sync.callback()
// and sync.arg() as parameters (i.e. invoking itself asynchronously), and then
// returns sync.WaitForResult().  The rest of the API can then be coded in
// standard asynchronous style.
template <class O>
class Sync final {
 public:
  void* arg() { return reinterpret_cast<void*>(this); }

  void (*callback())(void*, O) { return &Callback; }

  // Waits for the result of the call.
  O WaitForResult();

 private:
  // The callback function to use with Sync objects.
  // The argument should be the pointer to the Sync object.
  static void Callback(void* arg, O result);

  std::mutex mu_;
  std::condition_variable cv_;
  bool done_ = false;
  O result_;
};

template <class O>
void Sync<O>::Callback(void* arg, O result) {
  Sync<O>* sync = reinterpret_cast<Sync<O>*>(arg);

  std::lock_guard<std::mutex> lock{sync->mu_};
  assert(!sync->done_);

  sync->done_ = true;
  sync->result_ = result;

  // N.B. Since Sync<O> is stack-allocated, it's extremely important that the
  // last action taken be the release of the mutex; holding the mutex is the
  // only thing preventing the sync from being freed once done_ is set.  So
  // although normally, you'd want to drop the mutex and then call
  // notify_one(), in this particular case, we hold the mutex around the
  // notification call.
  sync->cv_.notify_one();
}

template <class O>
O Sync<O>::WaitForResult() {
  std::unique_lock<std::mutex> lock{mu_};
  cv_.wait(lock, [this] { return done_; });
  return result_;
}

}  // namespace vertexai
