// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "plaidml/op/op.h"

#include "plaidml_op_permute.hpp"
#include "plaidml_util.hpp"

void OpPermute::run(const plaidml::edsl::Tensor& I, plaidml::edsl::Tensor& O) {
  const auto& order = layer_->GetParamAsInts("order");
  auto layout = layer_->insData.front().lock()->getTensorDesc().getLayout();
  if (layout == NCHW) {
    // NB: Input data in the plugin is always in the NHWC format
    // If the IE has got NCHW, the axes are also for NCHW,
    // so we need to change them for NHWC case
    // after the transpositon input has got NCHW format
    // and we again put conversion from NCHW to NHWC
    std::vector<int64_t> to_nhwc = {0, 3, 1, 2};
    O = plaidml::op::transpose(I, Value{{Value(to_nhwc[order[0]]), Value(to_nhwc[order[1]]), Value(to_nhwc[order[2]]),
                                         Value(to_nhwc[order[3]])}});
    O = plaidml::op::transpose(O, Value{{Value{0}, Value{2}, Value{3}, Value{1}}});
  } else {
    std::vector<Value> values_vec;
    for (int i = 0; i < order.size(); ++i) {
      values_vec.push_back(Value(order[i]));
    }
    O = plaidml::op::transpose(I, Value(values_vec));
  }
}
