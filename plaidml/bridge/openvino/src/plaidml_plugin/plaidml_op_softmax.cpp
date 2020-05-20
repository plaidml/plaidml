// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional>
#include <vector>

#include "plaidml_op_softmax.hpp"
#include "plaidml_util.hpp"

#include "plaidml/op/op.h"

void OpSoftmax::run(const plaidml::edsl::Tensor& I, plaidml::edsl::Tensor& O) {
  auto* layer = dynamic_cast<SoftMaxLayer*>(layer_.get());
  const auto& layout = layer_->outData.front()->getTensorDesc().getLayout();

  std::vector<int64_t> to_nhwc = {0, 3, 1, 2};
  O = plaidml::op::softmax(I, layout == NCHW ? to_nhwc[layer->axis] : layer->axis);
}
