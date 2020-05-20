// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
//#include <regex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ie_blob.h>
#include <ie_common.h>
#include <ie_layers_property.hpp>
#include <ie_layouts.h>
#include <ie_precision.hpp>

#include "plaidml/core/ffi.h"
#include "plaidml/edsl/edsl.h"
#include "plaidml/exec/exec.h"

#include "any.hpp"

using namespace InferenceEngine;

#define REPLACE_WITH_STR(SRC, PATTERN, STR) SRC = std::regex_replace(SRC, std::regex(PATTERN), STR)
#define REPLACE_WITH_NUM(SRC, PATTERN, NUM) REPLACE_WITH_STR(SRC, PATTERN, std::to_string(NUM))
#define REPLACE_WITH_NUM_VECTOR(SRC, PATTERN, NUMS) \
  {                                                 \
    std::string result;                             \
    if (NUMS.size() > 0) {                          \
      result += std::to_string(NUMS[0]);            \
      for (int i = 1; i < NUMS.size(); i++) {       \
        result += "," + std::to_string(NUMS[i]);    \
      }                                             \
    }                                               \
    REPLACE_WITH_STR(SRC, PATTERN, result);         \
  }

namespace PlaidMLPlugin {

namespace util {
enum OpType : uint8_t {
  Convolution,
  Pooling,
  ScaleShift,
  ReLU,
  Eltwise,
  Reshape,
  Fc,
  Softmax,
  Power,
  Clamp,
  Permute,
  Concat,
  Sigmoid,
  Crop,
  Norm,
};

OpType str_to_type(const std::string& layer_type);
plaidml_datatype to_plaidml(Precision prec);
plaidml::edsl::LogicalShape to_plaidml(const TensorDesc& desc);
std::vector<int64_t> to_plaidml(const SizeVector& dims);
std::vector<int32_t> to_plaidml(const PropertyVector<unsigned int>& property);
plaidml::exec::Binding make_binding(const std::string& dev, const TensorDesc& desc);
Blob::Ptr make_shared_blob(const TensorDesc& desc);

void transpose(const uint8_t* src, const SizeVector& dims, const SizeVector& order, uint8_t* dst, size_t element_size);

std::string get_var_from_env(const std::string& var);
std::string find_device(const std::string& device_name);
std::string find_target(const std::string& device_name);

// It's more convenient wrapper over any
class Any {
 public:
  Any() = default;

  template <typename T>
  explicit Any(const T& v) : value(v) {}

  template <typename T>
  inline T& get() {
    return any_cast<typename std::remove_reference<T>::type>(value);
  }

  template <typename T>
  inline const T& get() const {
    return any_cast<typename std::remove_reference<T>::type>(value);
  }

 private:
  util::any value;
};

template <int... Ns>
struct sequence {};
template <int... Ns>
struct seq_gen;

template <int I, int... Ns>
struct seq_gen<I, Ns...> {
  using type = typename seq_gen<I - 1, I - 1, Ns...>::type;
};

template <int... Ns>
struct seq_gen<0, Ns...> {
  using type = sequence<Ns...>;
};

template <int N>
using make_sequence = typename seq_gen<N>::type;

template <std::size_t I, typename Target, typename First, typename... Remaining>
struct type_list_index_helper {
  static const constexpr bool is_same = std::is_same<Target, First>::value;
  static const constexpr std::size_t value =
      std::conditional<is_same, std::integral_constant<std::size_t, I>,
                       type_list_index_helper<I + 1, Target, Remaining...>>::type::value;
};

template <std::size_t I, typename Target, typename First>
struct type_list_index_helper<I, Target, First> {
  static_assert(std::is_same<Target, First>::value, "Type not found");
  static const constexpr std::size_t value = I;
};

template <typename Target, typename... Types>
struct type_list_index {
  static const constexpr std::size_t value = type_list_index_helper<0, Target, Types...>::value;
};

// FIXME Add check that all types are unique
template <typename... Ts>
class Magazine {
 public:
  template <typename T>
  using MapT = std::unordered_map<std::string, T>;

  template <typename T>
  MapT<T>& slot() {
    return std::get<type_list_index<T, Ts...>::value>(slots);
  }

  template <typename T>
  const MapT<T>& slot() const {
    return std::get<type_list_index<T, Ts...>::value>(slots);
  }

 private:
  std::tuple<MapT<Ts>...> slots;
};

}  // namespace util

}  // namespace PlaidMLPlugin
