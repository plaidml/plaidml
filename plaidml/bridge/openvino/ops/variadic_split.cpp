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

void registerVariadicSplit() {
  registerOp("VariadicSplit", [](const Context& ctx) {
    IE_ASSERT(ctx.operands.size() == 3);
    auto I = ctx.operands.at(0);
    auto axes = get_axis_vector_from_constant_operand(1, ctx.layer);
    IE_ASSERT(axes.size() == 1);
    auto axis = axes[0];
    auto split_lengths = cast_constant_operand<int32_t>(2, ctx.layer);

    auto ndims = I.rank();
    std::vector<edsl::TensorDim> I_dims(ndims);
    std::vector<edsl::TensorIndex> I_idxs(ndims);
    std::vector<edsl::Tensor> Os;
    I.bind_dims(I_dims);
    auto O_dims = I_dims;

    size_t split_size = 0;
    for (auto split : split_lengths) {
      if (split != -1) {
        split_size += split;
      }
    }
    auto placeholder = I_dims[axis] - split_size;

    edsl::TensorDim offset(0);
    for (auto split : split_lengths) {
      auto O_idxs = I_idxs;
      O_idxs[axis] = I_idxs[axis] - offset;
      if (split == -1) {
        O_dims[axis] = placeholder;
        offset = offset + placeholder;
      } else {
        O_dims[axis] = edsl::TensorDim(split);
        offset = offset + split;
      }
      Os.push_back(edsl::Contraction()  //
                       .outShape(O_dims)
                       .outAccess(O_idxs)
                       .assign(I(I_idxs)));
    }
    return edsl::make_tuple(Os);
  });
}

}  // namespace PlaidMLPlugin
