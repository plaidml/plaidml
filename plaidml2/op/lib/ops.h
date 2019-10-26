// Copyright 2019 Intel Corporation.

#pragma once

#include <string>
#include <unordered_map>

#include "plaidml2/edsl/edsl.h"

namespace plaidml {
namespace op {
namespace lib {

using Operation = std::function<edsl::Value(const edsl::Value& value)>;

class OperationRegistry {
 public:
  static OperationRegistry* Instance() {
    static OperationRegistry registry = BuildDefault();
    return &registry;
  }

  void Register(const std::string& name, Operation op) {  //
    registry_[name] = op;
  }

  const Operation Resolve(const std::string& name) {
    auto it = registry_.find(name);
    if (it == registry_.end()) {
      return nullptr;
    }
    return it->second;
  }

 private:
  static void AddDefaultOps(OperationRegistry* registry);

  static OperationRegistry BuildDefault() {
    OperationRegistry registry;
    AddDefaultOps(&registry);
    return registry;
  }

  std::unordered_map<std::string, Operation> registry_;
};

}  // namespace lib
}  // namespace op
}  // namespace plaidml
