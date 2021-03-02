// Copyright 2019 Intel Corporation.

#pragma once

#include <algorithm>
#include <string>
#include <unordered_map>
#include <utility>

#include "llvm/Support/FormatVariadic.h"

#include "plaidml/edsl/edsl.h"
#include "plaidml/op/op.h"

using namespace plaidml::edsl;  // NOLINT
using namespace plaidml::op;    // NOLINT

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

template <typename T>
std::pair<T, T> compute_padding_and_output_size(  //
    const T& input_size,                          //
    const T& filter_size,                         //
    int64_t stride,                               //
    plaidml::op::AutoPadMode autopad_mode,        //
    int64_t pad_lo,                               //
    int64_t pad_hi,                               //
    int64_t dilation,                             //
    int64_t data_dilation,                        //
    bool use_ceil_for_output_shape) {
  // Effective input and filter sizes are the sizes after dilations are
  // accounted for. So a 4x3 filter dilated by (3, 2) has an effective filter
  // size of 11 and 5 for its 2 spatial dims

  auto I_eff = (data_dilation * (input_size - 1)) + 1;  // Effective Input Size
  auto F_eff = (dilation * (filter_size - 1)) + 1;      // Effective Filter Size
  int64_t ceil_term =
      use_ceil_for_output_shape ? stride - 1 : 0;  // TODO: Will need to confirm that this is the intended behavior
  if (autopad_mode == AutoPadMode::EXPLICIT) {
    T pad_before(pad_lo);
    T output_size((I_eff + pad_lo + pad_hi - F_eff + stride + ceil_term) / stride);
    return std::pair<T, T>(pad_before, output_size);
  }
  if (autopad_mode == AutoPadMode::VALID) {
    T pad_before(0);
    T output_size((I_eff - F_eff + stride + ceil_term) / stride);
    return std::pair<T, T>(pad_before, output_size);
  }
  if (autopad_mode == AutoPadMode::SAME_LOWER || autopad_mode == AutoPadMode::SAME_UPPER) {
    T output_size((I_eff + stride - 1 + ceil_term) / stride);
    int64_t lower_term = (autopad_mode == AutoPadMode::SAME_LOWER) ? 1 : 0;
    T pad_before((max(0, (output_size - 1) * stride + F_eff - I_eff) + lower_term) / 2);
    return std::pair<T, T>(pad_before, output_size);
  }
  // throw std::runtime_error(llvm::formatv("Unexpected autopadding mode: {0}", to_string(autopad_mode)));
}

}  // namespace plaidml::op::lib
