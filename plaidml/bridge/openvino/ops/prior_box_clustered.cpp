// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset5.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;  // NOLINT[build/namespaces]
using ngraph::opset5::PriorBoxClustered;

namespace PlaidMLPlugin {

void registerPriorBoxClustered() {
  registerOp("PriorBoxClustered", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset5::PriorBoxClustered>(ctx.layer);
    IE_ASSERT(layer);
    IE_ASSERT(ctx.operands.size() == 2);
    // According to the size of output, the size of variance shall be restricted
    const ngraph::op::PriorBoxClusteredAttrs& attrs = layer->get_attrs();
    IE_ASSERT(attrs.variances.size() == 1 || attrs.variances.size() == 4 || attrs.variances.empty());

    auto precision = to_plaidml(ctx.layer->get_output_element_type(0));
    int64_t num_priors = static_cast<int64_t>(attrs.widths.size());
    auto variances = attrs.variances;
    if (variances.empty()) variances.push_back(0.1f);

    auto input0_shape = cast_constant_operand<int64_t>(0, layer);
    auto input1_shape = cast_constant_operand<int64_t>(1, layer);
    int64_t layer_height = input0_shape[0];
    int64_t layer_width = input0_shape[1];
    int64_t img_height = input1_shape[0];
    int64_t img_width = input1_shape[1];

    float step_w = attrs.step_widths;
    float step_h = attrs.step_heights;

    // TODO: Ref implementation in openvino currently not open "img_h && img_w && step" attributes
    // Uncomment once those been supported
    // if(attrs.img_h ! = 0)
    //   img_height = attrs.img_h;
    // if(attrs.img_w != 0)
    //   img_width = attrs.img_w;
    // if(attrs.step_heights == 0)
    //   step_h = attrs.step;
    // if(attrs.step_widths == 0)
    //   step_w = attrs.step;

    if (step_w == 0 && step_h == 0) {
      step_w = static_cast<float>(img_width) / layer_width;
      step_h = static_cast<float>(img_height) / layer_height;
    }

    // Create mask
    TensorShape shape_mask(precision, {1, 1, 1, 4});
    std::vector<float> cw_mask = {1, 0, 1, 0};
    Buffer buffer_cwm(shape_mask);
    buffer_cwm.copy_from(cw_mask.data());
    auto CW_mask = edsl::Constant(buffer_cwm, "cw_mask");

    std::vector<float> ch_mask = {0, 1, 0, 1};
    Buffer buffer_chm(shape_mask);
    buffer_chm.copy_from(ch_mask.data());
    auto CH_mask = edsl::Constant(buffer_chm, "ch_mask");

    std::vector<float> bw_mask = {-1, 0, 1, 0};
    Buffer buffer_bwm(shape_mask);
    buffer_bwm.copy_from(bw_mask.data());
    auto BW_mask = edsl::Constant(buffer_bwm, "bw_mask");

    std::vector<float> bh_mask = {0, -1, 0, 1};
    Buffer buffer_bhm(shape_mask);
    buffer_bhm.copy_from(bh_mask.data());
    auto BH_mask = edsl::Constant(buffer_bhm, "bh_mask");

    // Create box center
    auto CW = edsl::cast(
        edsl::index({edsl::TensorDim(layer_height), edsl::TensorDim(layer_width), edsl::TensorDim(1)}, 1), precision);
    auto CH = edsl::cast(
        edsl::index({edsl::TensorDim(layer_height), edsl::TensorDim(layer_width), edsl::TensorDim(1)}, 0), precision);
    auto CW_normalized = edsl::reshape((CW + attrs.offset) * step_w / img_width, {layer_height, layer_width, 1, 1});
    auto CH_normalized = edsl::reshape((CH + attrs.offset) * step_h / img_height, {layer_height, layer_width, 1, 1});

    // Create box shift
    TensorShape shape_box(precision, {1, 1, num_priors, 1});
    Buffer buffer_bw(shape_box);
    buffer_bw.copy_from(attrs.widths.data());
    edsl::Tensor BW = edsl::Constant(buffer_bw, "bw") / 2.0f / img_width;

    Buffer buffer_bh(shape_box);
    buffer_bh.copy_from(attrs.heights.data());
    edsl::Tensor BH = edsl::Constant(buffer_bh, "bh") / 2.0f / img_height;

    edsl::Tensor C_out = CW_normalized * CW_mask + CH_normalized * CH_mask + BW * BW_mask + BH * BH_mask;
    if (attrs.clip) C_out = op::clip(C_out, edsl::Tensor(0.0f), edsl::Tensor(1.0f));

    // Use variance as another channel
    int channel_size = layer_height * layer_width * num_priors * 4;
    edsl::Tensor Varian;
    if (variances.size() == 1) {
      auto Val = edsl::cast(edsl::Tensor(variances[0]), precision);
      Varian = op::broadcast(Val, {1, channel_size}, {});
    } else {
      TensorShape shape_va(precision, {1, 4});
      Buffer buffer_va(shape_va);
      buffer_va.copy_from(variances.data());
      auto V = edsl::Constant(buffer_va, "variances");
      Varian = edsl::reshape(op::repeat(V).count(channel_size / 4).axis(0), {1, channel_size});
    }

    C_out = op::concatenate({edsl::reshape(C_out, {1, channel_size}), Varian}, 0);
    return edsl::make_tuple(C_out);
  });
}

}  // namespace PlaidMLPlugin
