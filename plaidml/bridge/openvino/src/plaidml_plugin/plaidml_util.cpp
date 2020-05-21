// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_util.hpp"

#include "plaidml/core/core.h"
#include "plaidml/edsl/edsl.h"

using plaidml::edsl::LogicalShape;
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

plaidml::DType to_plaidml(Precision prec) {
  switch (prec) {
    case Precision::FP16:
      return plaidml::DType::FLOAT16;
    case Precision::FP32:
      return plaidml::DType::FLOAT32;
    case Precision::U8:
      return plaidml::DType::UINT8;
    case Precision::U16:
      return plaidml::DType::UINT16;
    case Precision::I8:
      return plaidml::DType::INT8;
    case Precision::I16:
      return plaidml::DType::INT16;
    case Precision::I32:
      return plaidml::DType::INT32;
    case Precision::I64:
      return plaidml::DType::INT64;
    default:
      THROW_IE_EXCEPTION << "Unsupported precision";
  }
}

LogicalShape to_plaidml(const TensorDesc& desc) {
  auto dtype = to_plaidml(desc.getPrecision());
  auto dims = to_plaidml(desc.getDims());
  return LogicalShape{dtype, dims};
}

std::vector<int64_t> to_plaidml(const SizeVector& dims) { return std::vector<int64_t>{dims.begin(), dims.end()}; }

}  // namespace PlaidMLPlugin
