// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset3.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;  // NOLINT[build/namespaces]
using ngraph::opset3::Bucketize;

namespace PlaidMLPlugin {

static OpRegistration reg("Bucketize", [](const Context& ctx) {
  auto* layer = ngraph::as_type<Bucketize>(ctx.layer);
  IE_ASSERT(ctx.operands.size() == 2);
  auto A = ctx.operands.at(0);
  auto B = ctx.operands.at(1);
  IE_ASSERT(B.rank() == 1);
  std::vector<edsl::TensorDim> B_dims(B.rank());
  B.bind_dims(B_dims);
  auto broadcast_result = op::repeat(op::unsqueeze(A, {-1})).count(B_dims[0]).axis(-1);
  auto output_type = to_plaidml(layer->get_output_type());
  auto one = edsl::cast(edsl::Tensor(1), output_type);
  auto zero = edsl::cast(edsl::Tensor(0), output_type);
  auto C = layer->get_with_right_bound() ? op::sum(select(broadcast_result > B, one, zero), edsl::make_tuple(-1))
                                         : op::sum(select(broadcast_result >= B, one, zero), edsl::make_tuple(-1));
  return edsl::make_tuple(C);
});

}  // namespace PlaidMLPlugin
