#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <utility>

namespace vertexai {

// A helper base class that allows you to derive from a it and then call 'make' to get
// an interned shared_ptr version of the object.  Usage is:
// class SomeClass : public Interned<SomeClass> {
//   SomeClass(int x) { x_ = x; }
//   bool operator<(const SomeClass& rhs) { return x_ < rhs.x_; }
// };
//
// std::shared_ptr<SomeClass> a = SomeClass::make(5)  // Forwards the constructor, makes a new object
// std::shared_ptr<SomeClass> b = SomeClass::make(6)  // Makes a new object
// std::shared_ptr<SomeClass> c = SomeClass::make(5)  // Finds an identical object in intern table, reuses
// assert(a != b)
// assert(a == b)  // Pointer equivience!

template <typename T>
struct Interned {
  template <typename... Args>
  static std::shared_ptr<T> make(const Args&... args) {
    typedef std::tuple<Args...> tuple_t;

    // N.B. We use a recursive mutex here so that the deleter can be invoked while we hold the internment mutex.
    static std::recursive_mutex interned_mu;
    static std::map<tuple_t, std::pair<unsigned, std::weak_ptr<T>>> interned;

    tuple_t key(args...);
    std::lock_guard<std::recursive_mutex> lock{interned_mu};

    // N.B. There may already be an entry in the map for this value.
    //
    // Here's a fun scenario:
    //   One thread inserts an entry
    //   The refcount on that entry's value drops to zero, causing its deleter to be run
    //   Before the deleter runs:
    //     Another thread goes to insert the same value
    //     The second thread gets the lock, inserts the value, and drops the lock
    //     The refcount on the *new* value goes to zero, causing *that* value's deleter to be run
    //
    // So now we have two deleters running -- sequentially, because of the mutex, but they are both poised to run.
    // If both deleters hold an iterator to the same map entry, we really need to make sure only one erases it.
    //
    // The two good ways to handle this are:
    //   * Use a multimap, so that each deleter gets its very own iterator.
    //   * Use a refcount in the map entry.
    //
    // We go with the refcount approach -- each incarnation of a value has its own deleter, and the
    // deleters manipulate the refcount to determine how many other deleters reference the entry.
    // (The code winds up being slightly simpler than dealing with a multimap.)

    // N.B. It doesn't matter whether we insert an entry or whether there was a pre-existing entry, because the weak
    // pointer within an existing entry may have been cleared.  So we do the insert, and attempt to lock the weak
    // pointer; if we succeed, we have something to return, and if we don't, then we need to allocate the object
    // regardless of whether or not the entry already existed in the map.
    auto it = interned.insert(std::make_pair(key, std::make_pair(0, std::weak_ptr<T>()))).first;
    std::shared_ptr<T> result = it->second.second.lock();
    if (result) {
      return result;
    }

    auto cleanup = [it](T * t) noexcept {
      delete t;
      std::lock_guard<std::recursive_mutex> lock{interned_mu};
      if (!--it->second.first) {
        interned.erase(it);
      }
    };

    // We start by allocating a unique_ptr with the correct deleter.  Unlike shared_ptr construction, this constructor
    // cannot throw, making it safe to use a bare new to allocate T.
    std::unique_ptr<T, decltype(cleanup)> uniq_result{new T{args...}, std::move(cleanup)};

    // At this point, the deleter *will* run at some point, so increment the refcount on the map entry:
    ++it->second.first;

    // Now, transfer that unique_ptr together with its deleter to the result shared_ptr.  This assignment *can* throw,
    // but that's okay; if it does, then before we drop the lock on the internment map, we will destroy the unique_ptr
    // holding the result, which will delete the result and erase it from the map.  (Note that assignment is guaranteed
    // to either succeed, or to leave the supplied unique_ptr intact.)
    result = std::move(uniq_result);

    // Finally, assign the map entry.
    it->second.second = result;

    return result;
  }
};

}  // namespace vertexai
