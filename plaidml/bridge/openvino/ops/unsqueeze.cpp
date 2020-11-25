// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset1.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

void registerUnsqueeze() {
  registerOp("unsqueeze", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 2);
    auto I = ctx.operands.at(0);
    auto axes = get_axis_set_from_constant_operand(1, ctx.layer);
    std::vector<edsl::TensorDim> I_dims(I.rank());
    size_t new_rank = I.rank() + axes.size();
    I.bind_dims(I_dims);
    std::vector<edsl::TensorDim> O_dims;
    size_t src_loc = 0;
    for (size_t dst_loc = 0; dst_loc < new_rank; dst_loc++) {
      if (axes.count(dst_loc)) {
        O_dims.push_back(edsl::TensorDim(1));
      } else {
        O_dims.push_back(I_dims[src_loc]);
        src_loc++;
      }
    }
    IE_ASSERT(src_loc == I.rank());  // Need to have used exactly all the input dims
    return edsl::make_tuple(op::reshape(I, edsl::make_tuple(O_dims)));
  });
}

}  // namespace PlaidMLPlugin
