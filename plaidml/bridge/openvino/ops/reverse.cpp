// Copyright (C) 2021 Intel Corporation
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

// TODO: Remove and replace use with get_axis_set_from_constant_operand once upstream fixed for negatives
template <typename T>
std::vector<T> cast_constant_operand(size_t operand_idx, ngraph::Node* layer) {
  auto* ngraph_const = ngraph::as_type<ngraph::op::Constant>(layer->get_input_node_ptr(operand_idx));
  if (ngraph_const) {
    return ngraph_const->cast_vector<T>();
  } else {
    THROW_IE_EXCEPTION << " input [1] is Unsupported inputType; ";
  }
}

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

void registerReverse() {
  registerOp("reverse", [](const Context& ctx) {
    auto* layer = ngraph::as_type<Reverse>(ctx.layer);
    IE_ASSERT(ctx.operands.size() == 2);
    auto I = ctx.operands.at(0);
    ngraph::AxisSet axes;
    if (layer->get_mode() == Reverse::Mode::INDEX) {
      std::vector<int64_t> raw_axes = cast_constant_operand<int64_t>(1, ctx.layer);
      for (auto ax : raw_axes) {
        if (ax < 0) {
          ax += I.rank();
          if (ax < 0) {
            THROW_IE_EXCEPTION << "Axis underflow in Reverse (requested axis more negative than rank of tensor)";
          }
        }
        axes.emplace(ax);
      }
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
}

}  // namespace PlaidMLPlugin
