// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset1.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

void registerSplit() {
  registerOp("split", [](const Context& ctx) {
    auto* layer = dynamic_cast<ngraph::opset1::Split*>(ctx.layer);
    IE_ASSERT(ctx.operands.size() == 2);
    auto I = ctx.operands.at(0);
    // operands.at(1) is unused, just read the Constant instead
    auto splits = layer->get_num_splits();
    auto axes = get_axis_vector_from_constant_operand(1, ctx.layer);
    IE_ASSERT(axes.size() == 1);
    auto axis = axes[0];

    auto ndims = I.rank();
    std::vector<edsl::TensorDim> I_dims(ndims);
    std::vector<edsl::TensorIndex> I_idxs(ndims);
    std::vector<edsl::Tensor> Os;
    I.bind_dims(I_dims);
    auto O_dims = I_dims;
    auto split_size = I_dims[axis] / splits;
    O_dims[axis] = split_size;
    for (size_t i = 0; i < splits; i++) {
      auto O_idxs = I_idxs;
      O_idxs[axis] = I_idxs[axis] - i * split_size;
      Os.push_back(plaidml::edsl::Contraction().outShape(O_dims).outAccess(O_idxs).assign(I(I_idxs)));
    }
    return edsl::make_tuple(Os);
  });
}

}  // namespace PlaidMLPlugin
