// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <vector>

#include "plaidml_op_scaleshift.hpp"
#include "plaidml_util.hpp"

#include "plaidml/op/op.h"

using namespace plaidml::edsl;
using namespace plaidml::exec;

void OpScaleShift::LoadWeights(State* state) {
  scaleshift_layer_ = dynamic_cast<ScaleShiftLayer*>(layer_.get());

  const auto& weight = scaleshift_layer_->_weights;
  const auto& bias = scaleshift_layer_->_biases;

  IE_ASSERT(scaleshift_layer_->_broadcast == 0 && "Now supported only broadcast == 0 for ScaleShift");

  auto load_if_not_empty = [&](const Blob::Ptr& blob) {
    if (blob) {
      auto binding = util::make_binding(state->device(), blob->getTensorDesc());
      binding.buffer.copy_from(blob->buffer());
      auto& bindings = state->slot<std::vector<Binding>>()[layer_->name];
      bindings.push_back(std::move(binding));
    }
  };

  load_if_not_empty(weight);
  load_if_not_empty(bias);
}

void OpScaleShift::run(const plaidml::edsl::Tensor& I, const plaidml::edsl::Tensor& Scale,
                       const plaidml::edsl::Tensor& Shift, plaidml::edsl::Tensor& O) {
  size_t num_dims = layer_->insData[0].lock()->getDims().size();
  IE_ASSERT(num_dims == 4 && "Now supported only 4D tensors for ScaleShift");

  TensorDim N, C, H, W;
  TensorIndex n, c, h, w;

  I.bind_dims(N, H, W, C);

  Scale.bind_dims(C);
  Shift.bind_dims(C);

  O = TensorOutput(N, H, W, C);

  O(n, h, w, c) = I(n, h, w, c) * Scale(c);
  O(n, h, w, c) = O(n, h, w, c) + Shift(c);
}
