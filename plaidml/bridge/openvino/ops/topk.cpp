// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset4.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]
using Direction = edsl::SortDirection;
using Mode = ngraph::opset4::TopK::Mode;
using Sort = ngraph::opset4::TopK::SortType;

namespace PlaidMLPlugin {

void registerTopK() {
  registerOp("topk", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset4::TopK>(ctx.layer);
    IE_ASSERT(ctx.operands.size() == 2);
    auto I = ctx.operands.at(0);
    auto K = layer->get_k();
    auto axis = layer->get_axis();
    auto mode = layer->get_mode();
    auto sort = layer->get_sort_type();
    long ndims = I.rank();

    Direction direction;
    switch (mode) {
      case Mode::MAX:
        direction = Direction::DESC;
        break;
      case Mode::MIN:
        direction = Direction::ASC;
        break;
      default:
        THROW_IE_EXCEPTION << "invalid topk sort mode";
    }

    edsl::Tensor values = op::sort(I, axis, direction);
    edsl::Tensor indices = edsl::argsort(I, axis, direction);
    edsl::Tensor idxs_topk = edsl::gather(indices, edsl::index({edsl::TensorDim(K)}, 0)).axis(axis);
    std::vector<edsl::Tensor> Os;

    switch (sort) {
      case Sort::NONE:
        // TODO: According to OV specs, the behavior of Sort::NONE is undefined.
        THROW_IE_EXCEPTION << "SortType::NONE is not implemented";
        break;
      case Sort::SORT_INDICES: {
        std::vector<edsl::Tensor> comb_idxs;
        std::vector<edsl::TensorDim> idxs_dims(ndims);
        idxs_topk.bind_dims(idxs_dims);
        auto idxs_topk_sorted = op::sort(idxs_topk, axis, Direction::ASC);
        for (int i = 0; i < ndims; i++) {
          if (i == axis) {
            comb_idxs.push_back(op::unsqueeze(idxs_topk_sorted, {ndims}));
          } else {
            comb_idxs.push_back(op::unsqueeze(edsl::index(idxs_dims, i), {ndims}));
          }
        }
        auto idxs_nd = op::concatenate(comb_idxs, -1);
        auto vals_topk = edsl::gather(I, idxs_nd).mode(edsl::GatherMode::ND);
        Os.push_back(vals_topk);
        Os.push_back(idxs_topk_sorted);
      } break;
      case Sort::SORT_VALUES: {
        edsl::Tensor vals_topk_sorted = edsl::gather(values, edsl::index({edsl::TensorDim(K)}, 0)).axis(axis);
        Os.push_back(vals_topk_sorted);
        Os.push_back(idxs_topk);
      } break;
      default:
        THROW_IE_EXCEPTION << "invalid topk sort type";
    }

    return edsl::make_tuple(Os);
  });
}

}  // namespace PlaidMLPlugin
