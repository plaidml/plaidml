// Copyright 2017, Vertex.AI. CONFIDENTIAL

#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>

namespace vertexai {
namespace tile {
namespace lang {

// A Scope defines a mapping from identifiers to information associated with those identifiers within a local scope,
// using static nesting to perform lookups.
template <class V>
class Scope {
 public:
  Scope() {}
  explicit Scope(Scope<V>* parent) : parent_{parent} {}

  // Lookup the key in this scope or a parent scope.
  // Throws std::out_of_range if the key cannot be found.
  V& Lookup(const std::string& key) {
    auto it = info_.find(key);
    if (it != info_.end()) {
      return it->second;
    }
    if (parent_) {
      return parent_->Lookup(key);
    }
    throw std::out_of_range{"Undeclared reference: " + key};
  }

  const V& Lookup(const std::string& key) const {
    auto it = info_.find(key);
    if (it != info_.end()) {
      return it->second;
    }
    if (parent_) {
      return parent_->Lookup(key);
    }
    throw std::out_of_range{"Undeclared reference: " + key};
  }

  bool Defines(const std::string& key) const {
    auto it = info_.find(key);
    if (it != info_.end()) {
      return true;
    }
    return false;
  }

  // Bind the key in the current scope (regardless of whether it's bound in a parent scope).
  // Throws std::logic_error if the key is already bound in the current scope.
  void Bind(const std::string& key, const V& value) {
    auto result = info_.emplace(key, value);
    if (!result.second) {
      throw std::logic_error{"Duplicate binding discovered: " + key};
    }
  }

 private:
  Scope<V>* parent_ = nullptr;
  std::unordered_map<std::string, V> info_;
};

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
