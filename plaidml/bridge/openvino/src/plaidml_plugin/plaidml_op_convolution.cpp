// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <vector>

#include "plaidml/core/core.h"
#include "plaidml/edsl/edsl.h"
#include "plaidml/exec/exec.h"
#include "plaidml/op/op.h"

#include "plaidml_op_convolution.hpp"
#include "plaidml_util.hpp"

using namespace plaidml::edsl;
using namespace plaidml::exec;

void OpConvolution::LoadWeights(State* state) {
  conv_layer_ = dynamic_cast<ConvolutionLayer*>(layer_.get());

  const auto& weight = conv_layer_->_weights;
  const auto& bias = conv_layer_->_biases;

  size_t O = conv_layer_->outData.front()->getDims()[1];
  size_t I = conv_layer_->insData.front().lock()->getDims()[1] / conv_layer_->_group;
  size_t H = conv_layer_->_kernel[1];
  size_t W = conv_layer_->_kernel[0];

  if (weight) {  // Weighs and biases may be placed in input blobs
    TensorDesc desc(weight->getTensorDesc().getPrecision(), {H, W, I, O}, Layout::ANY);
    auto binding = util::make_binding(state->device(), desc);

    util::transpose(weight->buffer().as<uint8_t*>(), {O, I, H, W}, {2, 3, 1, 0},
                    reinterpret_cast<uint8_t*>(binding.buffer.mmap_discard().data()), weight->element_size());

    auto& bindings = state->slot<std::vector<Binding>>()[layer_->name];
    bindings.push_back(std::move(binding));
  }

  if (bias) {
    auto binding = util::make_binding(state->device(), bias->getTensorDesc());
    binding.buffer.copy_from(bias->buffer());

    auto& bindings = state->slot<std::vector<Binding>>()[layer_->name];
    bindings.push_back(std::move(binding));
  }
}

void OpConvolution::Execute() {
  if (ctx_.inputs_.size() == 3) {
    run(ctx_.inputs_.at(0).get<Tensor>(), ctx_.inputs_.at(1).get<Tensor>(), ctx_.inputs_.at(2).get<Tensor>(),
        *(ctx_.outputs_.at(0).get<Tensor*>()));
  } else if (ctx_.inputs_.size() == 2) {
    run(ctx_.inputs_.at(0).get<Tensor>(), ctx_.inputs_.at(1).get<Tensor>(), *(ctx_.outputs_.at(0).get<Tensor*>()));
  } else {
    THROW_IE_EXCEPTION << "Supported convolution with 2 or 3 input tensors";
  }
}

void OpConvolution::run(const plaidml::edsl::Tensor& I, const plaidml::edsl::Tensor& K, const plaidml::edsl::Tensor& B,
                        plaidml::edsl::Tensor& O) {
  const auto& out_shape = conv_layer_->outData.front()->getDims();

  /* FIXME: Now op::library does't support convolution with adding bias
   * and we can't add bias manually after op::convolution
   * So it's implemented as it was in TILE
   */
  TensorDim N, CI, HI, WI, CO, KH, KW;
  TensorIndex no, co, ho, wo, ci, kh, kw;

  I.bind_dims(N, HI, WI, CI / conv_layer_->_group);
  K.bind_dims(KH, KW, CI, CO);

  O = TensorOutput(N, out_shape[2], out_shape[3], CO);
  O(no, ho, wo, co) += I(no, conv_layer_->_stride[1] * ho + conv_layer_->_dilation[1] * kh - conv_layer_->_padding[1],
                         conv_layer_->_stride[0] * wo + conv_layer_->_dilation[0] * kw - conv_layer_->_padding[0],
                         ci + co * (conv_layer_->_group / out_shape[1])) *
                       K(kh, kw, ci, co);

  O(no, ho, wo, co) += O(no, ho, wo, co) + B(co);
}

void OpConvolution::run(const plaidml::edsl::Tensor& I, const plaidml::edsl::Tensor& K, plaidml::edsl::Tensor& O) {
  O = plaidml::op::convolution(                  //
      I,                                         // Input
      K,                                         // Kernel
      util::to_plaidml(conv_layer_->_stride),    // strides
      util::to_plaidml(conv_layer_->_dilation),  // dilations
      util::to_plaidml(conv_layer_->_dilation),  // data_dilations
      {},                                        // filter_shape
      conv_layer_->_group,                       // groups
      "explicit",                                // autopad_mode
      util::to_plaidml(conv_layer_->_padding),   // manual_padding
      "nxc",                                     // input_layout
      "xck",                                     // filter_layout
      "none",                                    // group_layout
      false,                                     // winograd_allowed
      "",                                        // name
      "ungrouped",                               // autogroup_mode
      "none",                                    // deriv_mode
      {});                                       // result_shape
}
