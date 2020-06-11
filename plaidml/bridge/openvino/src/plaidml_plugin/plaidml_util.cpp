// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_util.hpp"

#include "plaidml/edsl/edsl.h"

using plaidml::edsl::LogicalShape;
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

plaidml::DType to_plaidml(const ngraph::element::Type& ng_type) {
  switch (ng_type) {
    case ngraph::element::Type_t::f16:
      return plaidml::DType::FLOAT16;
    case ngraph::element::Type_t::f32:
      return plaidml::DType::FLOAT32;
    case ngraph::element::Type_t::f64:
      return plaidml::DType::FLOAT64;
    case ngraph::element::Type_t::i8:
      return plaidml::DType::INT8;
    case ngraph::element::Type_t::i16:
      return plaidml::DType::INT16;
    case ngraph::element::Type_t::i32:
      return plaidml::DType::INT32;
    case ngraph::element::Type_t::i64:
      return plaidml::DType::INT64;
    case ngraph::element::Type_t::u8:
      return plaidml::DType::UINT8;
    case ngraph::element::Type_t::u16:
      return plaidml::DType::UINT16;
    case ngraph::element::Type_t::u32:
      return plaidml::DType::UINT32;
    case ngraph::element::Type_t::u64:
      return plaidml::DType::UINT64;
    case ngraph::element::Type_t::u1:
    case ngraph::element::Type_t::boolean:
    case ngraph::element::Type_t::bf16:
    case ngraph::element::Type_t::undefined:
    case ngraph::element::Type_t::dynamic:
    default:
      // TODO: Verify these are the unsupported types
      THROW_IE_EXCEPTION << "Unsupported element type";
  }
}

plaidml::op::AutoPadMode to_plaidml(const ngraph::op::PadType& ng_type) {
  switch (ng_type) {
    case ngraph::op::PadType::EXPLICIT:
      return plaidml::op::AutoPadMode::EXPLICIT;
    case ngraph::op::PadType::SAME_LOWER:
      return plaidml::op::AutoPadMode::SAME_LOWER;
    case ngraph::op::PadType::SAME_UPPER:
      return plaidml::op::AutoPadMode::SAME_UPPER;
    case ngraph::op::PadType::VALID:
      return plaidml::op::AutoPadMode::VALID;
    default:
      THROW_IE_EXCEPTION << "Unsupported autopad type";
  }
}

}  // namespace PlaidMLPlugin
