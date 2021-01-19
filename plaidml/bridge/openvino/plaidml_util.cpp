// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_util.hpp"

#include "plaidml/edsl/edsl.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

ngraph::AxisSet get_axis_set_from_constant_operand(size_t operand_idx, ngraph::Node* layer) {
  auto* axis_ngraph_op = ngraph::as_type<ngraph::op::Constant>(layer->get_input_node_ptr(operand_idx));
  if (axis_ngraph_op) {
    return axis_ngraph_op->get_axis_set_val();
  } else {
    THROW_IE_EXCEPTION << "Dynamic axis not currently supported by PlaidML plugin";
  }
}

ngraph::AxisVector get_axis_vector_from_constant_operand(size_t operand_idx, ngraph::Node* layer) {
  auto* axis_ngraph_op = ngraph::as_type<ngraph::op::Constant>(layer->get_input_node_ptr(operand_idx));
  if (axis_ngraph_op) {
    return axis_ngraph_op->get_axis_vector_val();
  } else {
    THROW_IE_EXCEPTION << "Dynamic axis not currently supported by PlaidML plugin";
  }
}

plaidml::DType to_plaidml(const InferenceEngine::Precision& precision) {
  switch (precision) {
    case InferenceEngine::Precision::FP16:
      return plaidml::DType::FLOAT16;
    case InferenceEngine::Precision::FP32:
      return plaidml::DType::FLOAT32;
    case InferenceEngine::Precision::I8:
      return plaidml::DType::INT8;
    case InferenceEngine::Precision::I16:
      return plaidml::DType::INT16;
    case InferenceEngine::Precision::I32:
      return plaidml::DType::INT32;
    case InferenceEngine::Precision::I64:
      return plaidml::DType::INT64;
    case InferenceEngine::Precision::U8:
      return plaidml::DType::UINT8;
    case InferenceEngine::Precision::U16:
      return plaidml::DType::UINT16;
    case InferenceEngine::Precision::U32:
      return plaidml::DType::UINT8;
    case InferenceEngine::Precision::U64:
      return plaidml::DType::UINT16;
    case InferenceEngine::Precision::BOOL:
      return plaidml::DType::BOOLEAN;
    default:
      // TODO: Verify these are the unsupported types
      THROW_IE_EXCEPTION << "Unsupported element type";
  }
}

plaidml::DType to_plaidml(const ngraph::element::Type& type) {
  switch (type) {
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

plaidml::op::AutoPadMode to_plaidml(const ngraph::op::PadType& type) {
  switch (type) {
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

plaidml::op::PadMode to_plaidml(const ngraph::op::PadMode& type) {
  switch (type) {
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
  auto* shape_ngraph_op = ngraph::as_type<ngraph::op::Constant>(layer->get_input_node_ptr(operand_idx));
  if (shape_ngraph_op) {
    return shape_ngraph_op->get_shape_val();
  } else {
    THROW_IE_EXCEPTION << "Dynamic shapes not currently supported by PlaidML plugin";
  }
}

ngraph::Coordinate get_coords_from_constant_operand(size_t operand_idx, ngraph::Node* layer) {
  auto* coord_ngraph_op = ngraph::as_type<ngraph::op::Constant>(layer->get_input_node_ptr(operand_idx));
  if (coord_ngraph_op) {
    return coord_ngraph_op->get_coordinate_val();
  } else {
    THROW_IE_EXCEPTION << "Dynamic coordinates not currently supported by PlaidML plugin";
  }
}

edsl::Tensor clip_activation(const std::string& func_name, bool should_clip, float clip, const edsl::Tensor& T) {
  edsl::Tensor T_clipped;
  if (should_clip) {
    T_clipped = op::clip(T, edsl::Tensor(-clip), edsl::Tensor(clip));
  } else {
    T_clipped = T;
  }
  if (func_name == "relu") {
    return op::relu(T_clipped);
  } else if (func_name == "sigmoid") {
    return op::sigmoid(T_clipped);
  } else if (func_name == "tanh") {
    return edsl::tanh(T_clipped);
  } else {
    THROW_IE_EXCEPTION << "Unsupported activation function";
  }
}

}  // namespace PlaidMLPlugin
