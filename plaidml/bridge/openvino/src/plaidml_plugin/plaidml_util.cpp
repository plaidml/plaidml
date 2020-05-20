// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_util.hpp"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <details/caseless.hpp>

#include "plaidml/core/core.h"
#include "plaidml/core/ffi.h"  // plaidml_datatype
#include "plaidml/edsl/edsl.h"

using namespace PlaidMLPlugin;

plaidml_datatype util::to_plaidml(Precision prec) {
  switch (prec) {
    case Precision::FP32:
      return PLAIDML_DATA_FLOAT32;
    case Precision::FP16:
      return PLAIDML_DATA_FLOAT16;
    case Precision::U8:
      return PLAIDML_DATA_UINT8;
    case Precision::U16:
      return PLAIDML_DATA_UINT16;
    case Precision::I8:
      return PLAIDML_DATA_INT8;
    case Precision::I16:
      return PLAIDML_DATA_INT16;
    case Precision::I32:
      return PLAIDML_DATA_INT32;
    case Precision::I64:
      return PLAIDML_DATA_INT64;
    default:
      THROW_IE_EXCEPTION << "Unsupported precision";
  }
}

Blob::Ptr util::make_shared_blob(const TensorDesc& desc) {
  const auto p = desc.getPrecision();

#define CASE(prec) \
  case prec:       \
    return InferenceEngine::make_shared_blob<PrecisionTrait<prec>::value_type>(desc);

  switch (p) {
    CASE(Precision::FP32);
    CASE(Precision::FP16);
    CASE(Precision::Q78);
    CASE(Precision::U16);
    CASE(Precision::U8);
    CASE(Precision::I8);
    CASE(Precision::BOOL);
    CASE(Precision::I32);
    CASE(Precision::I64);
    CASE(Precision::BIN);

    default:
      THROW_IE_EXCEPTION << "The plugin does not support input " << p.name() << " precision";
  }
}

/*


plaidml::exec::Binding util::make_binding(const std::string& dev, const TensorDesc& desc) {
  auto ph = Placeholder(util::to_plaidml(desc));

  // LogicalShape shape(ffi::call<plaidml_logical_shape*>(plaidml_logical_shape_clone, arg.shape));
  // TensorShape tensor_shape(shape.dtype(), shape.sizes());

  auto buff = plaidml::Buffer(dev, {ph.shape().dtype(), ph.shape().sizes()});

  return plaidml::exec::Binding{std::move(ph), std::move(buff)};

  /*Tensor tensor(ffi::call<plaidml_expr*>(plaidml_expr_clone, arg.tensor));
      LogicalShape shape(ffi::call<plaidml_logical_shape*>(plaidml_logical_shape_clone, arg.shape));
      ProgramArgument programArg{arg.is_input, tensor, shape, nullptr};
      if (arg.buffer) {
        TensorShape tensor_shape(shape.dtype(), shape.sizes());
        auto bufptr = ffi::call<plaidml_buffer*>(plaidml_buffer_clone, arg.buffer);
        programArg.buffer = std::make_shared<Buffer>(bufptr, tensor_shape);
      }


* /

    /*
    LogicalShape(DType dtype, const std::vector<int64_t>& dims)
          : ptr_(details::make_ptr(ffi::call<plaidml_logical_shape*>(
                plaidml_logical_shape_alloc, static_cast<plaidml_datatype>(dtype), dims.size(), dims.data()))) {}
    */

plaidml::edsl::LogicalShape util::to_plaidml(const TensorDesc& desc) {
  return {static_cast<plaidml::DType>(util::to_plaidml(desc.getPrecision())), util::to_plaidml(desc.getDims())};
}

std::vector<int64_t> util::to_plaidml(const SizeVector& dims) { return std::vector<int64_t>{dims.begin(), dims.end()}; }

std::vector<int32_t> util::to_plaidml(const PropertyVector<unsigned int>& property) {
  std::vector<int32_t> property_vec;
  for (int i = 0; i < property.size(); ++i) {
    property_vec.push_back(property[i]);
  }
  return property_vec;
}

util::OpType util::str_to_type(const std::string& layer_type) {
  std::unordered_map<std::string, util::OpType, details::CaselessHash<std::string>, details::CaselessEq<std::string>>
      type_table = {
          {"convolution", util::OpType::Convolution},
          {"pooling", util::OpType::Pooling},
          {"scaleshift", util::OpType::ScaleShift},
          {"relu", util::OpType::ReLU},
          {"eltwise", util::OpType::Eltwise},
          {"reshape", util::OpType::Reshape},
          {"innerproduct", util::OpType::Fc},
          {"fullyconnected", util::OpType::Fc},
          {"softmax", util::OpType::Softmax},
          {"power", util::OpType::Power},
          {"clamp", util::OpType::Clamp},
          {"permute", util::OpType::Permute},
          {"concat", util::OpType::Concat},
          {"sigmoid", util::OpType::Sigmoid},
          {"crop", util::OpType::Crop},
          {"norm", util::OpType::Norm},
          {"lrn", util::OpType::Norm},
      };

  auto it = type_table.find(layer_type);
  if (it == type_table.end()) THROW_IE_EXCEPTION << "Unsupported layer type : " << layer_type;
  return it->second;
}

void util::transpose(const uint8_t* src, const SizeVector& dims, const SizeVector& order, uint8_t* dst,
                     size_t element_size) {
  const size_t kNumDims = dims.size();
  IE_ASSERT(kNumDims == 4);

  SizeVector in_strides;
  in_strides.reserve(kNumDims);
  size_t total_stride = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>()) * element_size;

  for (int i = 0; i < kNumDims; ++i) {
    total_stride /= dims[i];
    in_strides.push_back(total_stride);
  }

  SizeVector out_dims;
  out_dims.reserve(kNumDims);
  for (int i = 0; i < kNumDims; ++i) {
    out_dims.push_back(dims[order[i]]);
  }

  size_t dst_i = 0;
  for (int i0 = 0; i0 < out_dims[0]; ++i0) {
    for (int i1 = 0; i1 < out_dims[1]; ++i1) {
      for (int i2 = 0; i2 < out_dims[2]; ++i2) {
        for (int i3 = 0; i3 < out_dims[3]; ++i3) {
          size_t src_i = in_strides[order[0]] * i0 + in_strides[order[1]] * i1 + in_strides[order[2]] * i2 +
                         in_strides[order[3]] * i3;

          for (int i = 0; i < element_size; ++i) {
            dst[dst_i++] = src[src_i + i];
          }
        }
      }
    }
  }
}

std::string util::get_var_from_env(const std::string& var) {
  std::string str = plaidml::ffi::str(plaidml::ffi::call<plaidml_string*>(plaidml_settings_get, var.c_str()));
  if (str.empty()) {
    THROW_IE_EXCEPTION << "Environment variable " << var.c_str() << " is empty";
  }
  return str;
}

std::string util::find_device(const std::string& configuration_type) {
  // FIXME: remove code duplication for find_device() and find_target in the future
  auto devices = plaidml::exec::list_devices();
  if (devices.empty()) {
    THROW_IE_EXCEPTION << "Unable to find any PlaidML devices";
  }
  /* If we get configuration_type then search device in devices list
   * If configuration_type is empty then we look the environment variable and put names from environment variable
   * If environment variable is empty then throw exception
   */
  if (!configuration_type.empty()) {
    auto dit = std::find_if(devices.begin(), devices.end(), [&configuration_type](const std::string& curr_name) {
      return ((configuration_type == "cpu" && curr_name.find("cpu") != std::string::npos) ||
              (configuration_type == "gpu" && curr_name.find("hd_graphics") != std::string::npos) ||
              (configuration_type == "cuda" && curr_name.find("nvidia") != std::string::npos));
    });
    if (dit == devices.end()) {
      THROW_IE_EXCEPTION << "Device " << configuration_type << " not found ";
    }
    return *dit;
  } else {
    return get_var_from_env("PLAIDML_DEVICE");
  }
}
std::string util::find_target(const std::string& configuration_type) {
  auto targets = plaidml::edsl::list_targets();
  if (targets.empty()) {
    THROW_IE_EXCEPTION << "Unable to find any PlaidML targets";
  }
  /* If we get configuration_type then search target in targets list
   * If configuration_type is empty then we look the environment variable and put names from environment variable
   * If environment variable is empty then throw exception
   */
  if (!configuration_type.empty()) {
    auto dit = std::find_if(targets.begin(), targets.end(), [&configuration_type](const std::string& curr_name) {
      return ((configuration_type == "cpu" && curr_name.find("cpu") != std::string::npos) ||
              (configuration_type == "gpu" && curr_name.find("intel_") != std::string::npos &&
               curr_name.find("_opencl") != std::string::npos) ||
              (configuration_type == "cuda" && curr_name.find("nvidia") != std::string::npos));
    });
    if (dit == targets.end()) {
      THROW_IE_EXCEPTION << "Target for " << configuration_type << " not found ";
    }
    return *dit;
  } else {
    return get_var_from_env("PLAIDML_TARGET");
  }
}
