// Copyright 2019 Intel Corporation.

#pragma once

#include <string>
#include <unordered_map>

#include "plaidml/edsl/edsl.h"
#include "plaidml/op/op.h"

namespace plaidml::op::lib {

using Operation = std::function<edsl::Value(const edsl::Value& value)>;

class OperationRegistry {
 public:
  static OperationRegistry* Instance() {
    static OperationRegistry registry;
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
  std::unordered_map<std::string, Operation> registry_;
};

void RegisterOps();

std::pair<edsl::TensorDim, edsl::TensorDim> compute_padding_and_output_size(  //
    const edsl::TensorDim& input_size,                                        //
    const edsl::TensorDim& filter_size,                                       //
    int64_t stride,                                                           //
    AutoPadMode autopad_mode,                                                 //
    int64_t pad_lo,                                                           //
    int64_t pad_hi,                                                           //
    int64_t dilation,                                                         //
    int64_t data_dilation,                                                    //
    bool use_ceil_for_output_shape);

}  // namespace plaidml::op::lib
