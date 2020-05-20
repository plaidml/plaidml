// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_op_pooling.hpp"
#include "plaidml_util.hpp"

#include "plaidml/op/op.h"

void OpPooling::run(const plaidml::edsl::Tensor& I, plaidml::edsl::Tensor& O) {
  auto* layer = dynamic_cast<PoolingLayer*>(layer_.get());
  const auto& out_shape = layer->outData.front()->getDims();

  int kernel_h = layer->_kernel[1];
  int kernel_w = layer->_kernel[0];
  int stride_h = layer->_stride[1];
  int stride_w = layer->_stride[0];
  int pad_h = layer->_padding[1];
  int pad_w = layer->_padding[0];

  auto type_to_str = [&](PoolingLayer::PoolType type) {
    switch (type) {
      case PoolingLayer::PoolType::MAX:
        return "max";
      case PoolingLayer::PoolType::AVG:
        return "avg";
      default:
        THROW_IE_EXCEPTION << "PoolType " << type << " isn't supported ";
    }
  };

  auto type = layer->GetParamAsString("rounding_type", "");

  O = plaidml::op::pool(I, type_to_str(layer->_type), {kernel_h, kernel_w}, {stride_h, stride_w}, "none",
                        {pad_h, pad_w}, "nhwc", !layer->_exclude_pad, type == "ceil");
}
