// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>

#include <boost/optional.hpp>

namespace vertexai {
namespace tile {
namespace lang {

// A Scope defines a mapping from identifiers to information associated with those identifiers within a local scope,
// using static nesting to perform lookups.
template <class V>
class Scope {
 public:
  Scope() {}
  explicit Scope(const Scope<V>* parent) : parent_{parent} {}

  // Lookup the key in this scope or a parent scope.
  boost::optional<V> Lookup(const std::string& key) const {
    auto it = items_.find(key);
    if (it != items_.end()) {
      return it->second;
    }
    if (parent_) {
      return parent_->Lookup(key);
    }
    return boost::none;
  }

  bool Defines(const std::string& key) const {
    auto it = items_.find(key);
    if (it != items_.end()) {
      return true;
    }
    return false;
  }

  // Bind the key in the current scope (regardless of whether it's bound in a parent scope).
  // Throws std::logic_error if the key is already bound in the current scope.
  void Bind(const std::string& key, const V& value) {
    auto result = items_.emplace(key, value);
    if (!result.second) {
      throw std::logic_error{"Duplicate binding discovered: " + key};
    }
  }

 private:
  const Scope<V>* parent_ = nullptr;
  std::unordered_map<std::string, V> items_;
};

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
