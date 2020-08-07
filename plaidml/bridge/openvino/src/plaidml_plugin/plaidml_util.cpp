// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_util.hpp"

#include "ngraph/op/constant.hpp"

#include "plaidml/edsl/edsl.h"

using plaidml::edsl::LogicalShape;
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

ngraph::AxisSet get_axis_set_from_constant_operand(size_t operand_idx, ngraph::Node* layer) {
  auto axis_ngraph_op =
      std::dynamic_pointer_cast<ngraph::op::Constant>(layer->input_value(operand_idx).get_node_shared_ptr());
  if (axis_ngraph_op) {
    return axis_ngraph_op->get_axis_set_val();
  } else {
    THROW_IE_EXCEPTION << "Dynamic axis not currently supported by PlaidML plugin";
  }
}

ngraph::AxisVector get_axis_vector_from_constant_operand(size_t operand_idx, ngraph::Node* layer) {
  auto axis_ngraph_op =
      std::dynamic_pointer_cast<ngraph::op::Constant>(layer->input_value(operand_idx).get_node_shared_ptr());
  if (axis_ngraph_op) {
    return axis_ngraph_op->get_axis_vector_val();
  } else {
    THROW_IE_EXCEPTION << "Dynamic axis not currently supported by PlaidML plugin";
  }
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
    case ngraph::element::Type_t::boolean:
      return plaidml::DType::BOOLEAN;
    case ngraph::element::Type_t::u1:
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

plaidml::op::PadMode to_plaidml(const ngraph::op::PadMode& ng_type) {
  switch (ng_type) {
    case ngraph::op::PadMode::CONSTANT:
      return plaidml::op::PadMode::CONSTANT;
    case ngraph::op::PadMode::EDGE:
      return plaidml::op::PadMode::EDGE;
    case ngraph::op::PadMode::REFLECT:
      return plaidml::op::PadMode::REFLECT;
    case ngraph::op::PadMode::SYMMETRIC:
      return plaidml::op::PadMode::SYMMETRIC;
    default:
      THROW_IE_EXCEPTION << "Unsupported autopad type";
  }
}

ngraph::Shape get_shape_from_constant_operand(size_t operand_idx, ngraph::Node* layer) {
  auto shape_ngraph_op =
      std::dynamic_pointer_cast<ngraph::op::Constant>(layer->input_value(operand_idx).get_node_shared_ptr());
  if (shape_ngraph_op) {
    return shape_ngraph_op->get_shape_val();
  } else {
    THROW_IE_EXCEPTION << "Dynamic shapes not currently supported by PlaidML plugin";
  }
}

ngraph::Coordinate get_coords_from_constant_operand(size_t operand_idx, ngraph::Node* layer) {
  auto coord_ngraph_op =
      std::dynamic_pointer_cast<ngraph::op::Constant>(layer->input_value(operand_idx).get_node_shared_ptr());
  if (coord_ngraph_op) {
    return coord_ngraph_op->get_coordinate_val();
  } else {
    THROW_IE_EXCEPTION << "Dynamic coordinates not currently supported by PlaidML plugin";
  }
}

}  // namespace PlaidMLPlugin
