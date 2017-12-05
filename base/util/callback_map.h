
#pragma once

#include <map>

namespace vertexai {

// A helper class to provide a way for objects that need to stay alive while
// a callback is in-flight. PendingCallbackMap safely handles cases where
// a callback needs to be processed only once but might be invoked synchronously or asynchronously.
template <typename T>
class PendingCallbackMap {
 public:
  // Use Acquire to transfer ownership of the object that needs to stay alive while an event is in-flight.
  // The value returned is an opaque handle to the object.
  // It returns a void* so that its easy to pass into C-based APIs which typically take a
  // 'void* user_data' argument.
  // Use the returned handle in subsequent calls to Release().
  // This method is thread-safe.
  void* Acquire(std::unique_ptr<T> ptr) {
    std::lock_guard<std::mutex> lock{mu_};
    auto handle = counter_++;
    pending_.emplace(handle, std::move(ptr));
    return reinterpret_cast<void*>(handle);
  }

  // Release does a lookup on the handle and transfer the pending object to the caller if it is found.
  // Otherwise a blank unique_ptr is returned.
  // Release is designed to support multiple calls with the same handle.
  // This method is thread-safe.
  std::unique_ptr<T> Release(void* handle) {
    std::lock_guard<std::mutex> lock{mu_};
    std::unique_ptr<T> ptr;
    auto it = pending_.find(reinterpret_cast<size_t>(handle));
    if (it != pending_.end()) {
      // transfer ownership to the caller
      ptr.swap(it->second);
      pending_.erase(it);
    }
    return ptr;
  }

 private:
  std::mutex mu_;
  size_t counter_ = 0;
  std::map<size_t, std::unique_ptr<T>> pending_;
};

}  // namespace vertexai
