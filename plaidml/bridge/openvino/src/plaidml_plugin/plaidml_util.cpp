// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_util.hpp"

#include "ngraph/op/constant.hpp"

#include "plaidml/edsl/edsl.h"

using plaidml::edsl::LogicalShape;
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

ngraph::AxisSet get_axes_from_constant_operand(size_t operand_idx, ngraph::Node* layer) {
  ngraph::AxisSet axes;
  auto axis_ngraph_op =
      std::dynamic_pointer_cast<ngraph::op::Constant>(layer->input_value(operand_idx).get_node_shared_ptr());
  if (axis_ngraph_op) {
    axes = axis_ngraph_op->get_axis_set_val();
  } else {
    THROW_IE_EXCEPTION << "Dynamic axis not currently supported by PlaidML plugin";
  }
  return axes;
}

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

}  // namespace PlaidMLPlugin
