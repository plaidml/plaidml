// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <utility>
#include <vector>

#include "plaidml_op_concat.hpp"
#include "plaidml_util.hpp"

#include "plaidml/op/op.h"

/* NB: Concat doesn't have a single signature
 * because of the diffrent of inputs.
 * Therefore, collect them in a vector and pass them as a single input
 */
void OpConcat::PackInputs(State* state) {
  std::vector<Tensor> inputs;
  for (const auto in_data : layer_->insData) {
    const auto& name = in_data.lock()->getName();
    inputs.push_back(state->slot<plaidml::edsl::Tensor>()[name]);
  }

  ctx_.inputs_.emplace_back(std::move(inputs));
}

void OpConcat::run(const std::vector<plaidml::edsl::Tensor>& inputs, plaidml::edsl::Tensor& O) {
  auto* concat = dynamic_cast<ConcatLayer*>(layer_.get());
  const auto& layout = layer_->outData.front()->getTensorDesc().getLayout();

  std::vector<int64_t> to_nhwc = {0, 3, 1, 2};
  O = plaidml::op::concatenate(inputs, layout == NCHW ? to_nhwc[concat->_axis] : concat->_axis);
}
