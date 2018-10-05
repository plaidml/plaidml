// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <functional>
#include <list>
#include <map>
#include <mutex>

namespace vertexai {
namespace tile {

// A simplistic LRU cache implementation.
// The cache must be externally synchronized.
// Value must be CopyConstructible.
template <typename Key, typename Value, typename Compare = std::less<Key>>
class LruCache {
 public:
  explicit LruCache(std::size_t size_max) : size_max_(size_max) {}

  // Finds the value associated with the specified key.  If the value is not in
  // the cache, Builder() will be invoked to construct it; this adds a value to
  // the cache, which may result in an eviction of the least recently used
  // entry in the cache.  If an exception is thrown by the builder, the cache
  // is left unmodified, as if this function was never called (strong exception
  // guarantee).
  template <typename Builder>
  Value Lookup(const Key& key, const Builder& builder) {
    std::unique_lock<std::mutex> lock{mu_};

    auto it = entries_.find(key);

    if (it != entries_.end()) {
      lru_.erase(it->second.lru_ent);
      it->second.lru_ent = lru_.emplace(lru_.begin(), LruEnt{it});
      return it->second.value;
    }

    Value result = builder();
    if (size_max_ > 0) {
      it = entries_.emplace(key, MapEnt{result, lru_.end()}).first;
      while (lru_.size() > 0 && size_max_ < entries_.size()) {
        entries_.erase(lru_.back().map_ent);
        lru_.pop_back();
      }
      it->second.lru_ent = lru_.emplace(lru_.begin(), LruEnt{it});
    }
    return result;
  }

 private:
  struct LruEnt;

  struct MapEnt {
    Value value;
    typename std::list<LruEnt>::iterator lru_ent;
  };

  struct LruEnt {
    typename std::map<Key, MapEnt>::iterator map_ent;
  };

  // The maximum cache size.
  const std::size_t size_max_;

  // The map/LRU mutex.
  std::mutex mu_;

  // The entry lookup map.
  std::map<Key, MapEnt, Compare> entries_;

  // The LRU list.  Recently used entries are at the front; the next entry to
  // evict is at the back.
  std::list<LruEnt> lru_;
};

}  // namespace tile
}  // namespace vertexai
