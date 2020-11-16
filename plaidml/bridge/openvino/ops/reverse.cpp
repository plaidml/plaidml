// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset1.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;  // NOLINT[build/namespaces]
using ngraph::opset1::Reverse;

namespace {

ngraph::AxisSet cast_constant_axis_mask(size_t operand_idx, ngraph::Node* layer) {
  auto ngraph_const =
      std::dynamic_pointer_cast<ngraph::op::Constant>(layer->input_value(operand_idx).get_node_shared_ptr());
  if (!ngraph_const) {
    THROW_IE_EXCEPTION << "Dynamic axes not currently supported by PlaidML plugin.";
  }
  auto bool_mask = ngraph_const->cast_vector<bool>();
  ngraph::AxisSet axis_set{};
  for (size_t i = 0; i < static_cast<size_t>(bool_mask.size()); ++i) {
    if (bool_mask[i]) {
      axis_set.emplace(i);
    }
  }
  return axis_set;
}

}  // namespace

namespace PlaidMLPlugin {

static OpRegistration reg("reverse", [](const Context& ctx) {
  auto* layer = ngraph::as_type<Reverse>(ctx.layer);
  IE_ASSERT(ctx.operands.size() == 2);
  auto I = ctx.operands.at(0);
  ngraph::AxisSet axes;
  if (layer->get_mode() == Reverse::Mode::INDEX) {
    axes = get_axis_set_from_constant_operand(1, ctx.layer);
  } else if (layer->get_mode() == Reverse::Mode::MASK) {
    axes = cast_constant_axis_mask(1, ctx.layer);
  }

  std::vector<edsl::TensorDim> dims(I.rank());
  I.bind_dims(dims);
  std::vector<edsl::TensorIndex> I_idxs(I.rank());
  std::vector<edsl::TensorIndex> O_idxs;

  for (size_t axis = 0; axis < I.rank(); axis++) {
    if (axes.count(axis)) {
      O_idxs.push_back(dims[axis] - 1 - I_idxs[axis]);
    } else {
      O_idxs.push_back(I_idxs[axis]);
    }
  }

  return edsl::make_tuple(edsl::Contraction().outShape(dims).outAccess(O_idxs).assign(I(I_idxs)));
});

}  // namespace PlaidMLPlugin
