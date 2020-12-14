// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset4.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;  // NOLINT[build/namespaces]
using ngraph::opset4::PriorBox;

namespace PlaidMLPlugin {

class PriorBoxImpl {
 public:
  PriorBoxImpl(const Context& context, const ngraph::op::PriorBoxAttrs& attributes);

  // Fixed size path uses fixed_size, density, fixed_ratio | aspect_ratio
  void processFixedSizePath();

  // Min size path uses min_size, max_size, aspect_ratio, scale_all_sizes
  void processMinSizePath();

  // Append variance after former result
  void processVariance();

  edsl::Tensor& getOutput() { return C_out; }

 private:
  // Preprocess data with some attributes
  void prepareConfig();

  // Produce a tensor which data are interlaced with W and H
  // Use this to produce box center for different path
  void produceCenterBase();

  const Context& ctx;
  const ngraph::op::PriorBoxAttrs& attrs;
  DType precision;
  int H;
  int W;
  edsl::Tensor IH;
  edsl::Tensor IW;
  int num_of_prior;
  int min_element_size;
  edsl::Tensor Step;
  edsl::Tensor Step_x;
  edsl::Tensor Step_y;
  float offset;
  std::vector<float> aspect_ratios;
  std::vector<float> aspect_ratios_scale_size;
  std::vector<float> variance;
  edsl::Tensor Min_size;
  edsl::Tensor CW_normalized;
  edsl::Tensor CH_normalized;
  edsl::Tensor CW_mask;
  edsl::Tensor CH_mask;
  edsl::Tensor BW_mask;
  edsl::Tensor BH_mask;
  edsl::Tensor C_out;
};

PriorBoxImpl::PriorBoxImpl(const Context& context, const ngraph::op::PriorBoxAttrs& attributes)
    : ctx(context), attrs(attributes) {
  prepareConfig();
  produceCenterBase();
}

// Preprocess data with some attributes
void PriorBoxImpl::prepareConfig() {
  precision = to_plaidml(ctx.layer->get_output_element_type(0));
  auto* input_shape_ngraph_op = ngraph::as_type<ngraph::op::Constant>(ctx.layer->get_input_node_ptr(0));
  if (input_shape_ngraph_op == nullptr) {
    THROW_IE_EXCEPTION << "Dynamic output_size of PriorBox not currently supported by PlaidML plugin";
  }
  std::vector<int> input_shape = input_shape_ngraph_op->get_vector<int>();
  H = input_shape[0];
  W = input_shape[1];
  auto IHIW = ctx.operands.at(1);
  IH = edsl::cast(op::slice(IHIW).add_dims({0}), precision);
  IW = edsl::cast(op::slice(IHIW).add_dims({1}), precision);

  num_of_prior = ngraph::op::PriorBox::number_of_priors(attrs);

  aspect_ratios.push_back(1.0f);
  for (const auto& aspect_ratio : attrs.aspect_ratio) {
    bool exist = false;
    for (const auto existed_value : aspect_ratios) exist |= std::fabs(aspect_ratio - existed_value) < 1e-6;

    if (!exist) {
      aspect_ratios.push_back(aspect_ratio);
      if (attrs.flip) {
        aspect_ratios.push_back(1.0f / aspect_ratio);
      }
    }
  }
  for (float ar : aspect_ratios) {
    if (std::fabs(ar - 1.0f) < 1e-6) {
      continue;
    }
    aspect_ratios_scale_size.push_back(std::sqrt(ar));
  }

  variance = attrs.variance;
  if (variance.empty()) variance.push_back(0.1f);

  Step = edsl::cast(edsl::Tensor(attrs.step), precision);
  min_element_size = attrs.min_size.size();
  TensorShape shape_ms(precision, {1, 1, min_element_size, 1});
  Buffer buffer_ms(shape_ms);
  buffer_ms.copy_from(attrs.min_size.data());
  Min_size = edsl::Constant(buffer_ms, "min_size");
  if (!attrs.scale_all_sizes) {
    // mxnet-like PriorBox
    if (attrs.step == -1)
      Step = 1.f * IH / H;
    else
      Step = Step * IH;
    Min_size = Min_size * IH;
  }
  edsl::Tensor a = IW / W;
  edsl::Tensor b = Step;
  Step_x = edsl::select(Step == 0, IW / W, Step);
  Step_y = edsl::select(Step == 0, IH / H, Step);

  offset = attrs.offset;

  // Create mask
  TensorShape shape_mask(precision, {1, 1, 1, 4});
  std::vector<float> cw_mask = {1, 0, 1, 0};
  Buffer buffer_cwm(shape_mask);
  buffer_cwm.copy_from(cw_mask.data());
  CW_mask = edsl::Constant(buffer_cwm, "cw_mask");

  std::vector<float> ch_mask = {0, 1, 0, 1};
  Buffer buffer_chm(shape_mask);
  buffer_chm.copy_from(ch_mask.data());
  CH_mask = edsl::Constant(buffer_chm, "ch_mask");

  std::vector<float> bw_mask = {-1, 0, 1, 0};
  Buffer buffer_bwm(shape_mask);
  buffer_bwm.copy_from(bw_mask.data());
  BW_mask = edsl::Constant(buffer_bwm, "bw_mask");

  std::vector<float> bh_mask = {0, -1, 0, 1};
  Buffer buffer_bhm(shape_mask);
  buffer_bhm.copy_from(bh_mask.data());
  BH_mask = edsl::Constant(buffer_bhm, "bh_mask");
}

void PriorBoxImpl::produceCenterBase() {
  auto CW = edsl::cast(edsl::index({edsl::TensorDim(H), edsl::TensorDim(W), edsl::TensorDim(1)}, 1), precision);
  auto CH = edsl::cast(edsl::index({edsl::TensorDim(H), edsl::TensorDim(W), edsl::TensorDim(1)}, 0), precision);
  CW_normalized =
      edsl::reshape(edsl::select(Step == 0, (CW + 0.5f) * Step_x / IW, (CW + offset) * Step / IW), {H, W, 1, 1});
  CH_normalized =
      edsl::reshape(edsl::select(Step == 0, (CH + 0.5f) * Step_y / IH, (CH + offset) * Step / IH), {H, W, 1, 1});
}

void PriorBoxImpl::processFixedSizePath() {
  auto CW_mask_f =
      edsl::reshape(op::slice(CW_mask).add_dim(0, 1).add_dim(0, 1).add_dim(0, 1).add_dim(0, 2), {1, 1, 1, 1, 1, 2});
  auto CH_mask_f =
      edsl::reshape(op::slice(CH_mask).add_dim(0, 1).add_dim(0, 1).add_dim(0, 1).add_dim(0, 2), {1, 1, 1, 1, 1, 2});
  auto BW_mask_f =
      edsl::reshape(op::slice(BW_mask).add_dim(0, 1).add_dim(0, 1).add_dim(0, 1).add_dim(0, 2), {1, 1, 1, 1, 1, 2});
  auto BH_mask_f =
      edsl::reshape(op::slice(BH_mask).add_dim(0, 1).add_dim(0, 1).add_dim(0, 1).add_dim(0, 2), {1, 1, 1, 1, 1, 2});

  int fixed_size_count = attrs.fixed_size.size();
  int fixed_ratio_count = attrs.fixed_ratio.size();
  bool first_out = true;
  size_t fixed_size_s = 0;
  int density_s = 0, shift = 0;
  float var = 0;

  for (size_t s = 0; s < fixed_size_count; ++s) {
    fixed_size_s = static_cast<size_t>(attrs.fixed_size[s]);
    density_s = static_cast<int>(attrs.density[s]);
    shift = static_cast<int>(attrs.fixed_size[s] / attrs.density[s]);
    var = -(attrs.fixed_size[s] / 2) + shift / 2.f;

    // Create a density which last dimenstion is 2 (for center data which are WH)
    edsl::TensorDim e(density_s), one(1);

    auto Density_w = edsl::reshape((edsl::cast(edsl::index({e, e, one}, 1), precision) * shift + var) / IW,
                                   {1, 1, 1, density_s, density_s, 1});
    auto Density_h = edsl::reshape((edsl::cast(edsl::index({e, e, one}, 0), precision) * shift + var) / IH,
                                   {1, 1, 1, density_s, density_s, 1});

    if (fixed_ratio_count) {
      std::vector<float> fixed_ratio_ar;
      for (auto ar : attrs.fixed_ratio) {
        fixed_ratio_ar.push_back(std::sqrt(ar));
      }
      TensorShape shape_fr(precision, {1, 1, fixed_ratio_count, 1, 1, 1});
      Buffer buffer_fr(shape_fr);
      buffer_fr.copy_from(fixed_ratio_ar.data());
      auto Fixed_ratio = edsl::Constant(buffer_fr, "fixed_ratio");

      // Expand center for output
      auto CW_d = edsl::reshape(CW_normalized, {H, W, 1, 1, 1, 1}) + Density_w;
      auto CH_d = edsl::reshape(CH_normalized, {H, W, 1, 1, 1, 1}) + Density_h;
      auto Center = CW_d * CW_mask_f + CH_d * CH_mask_f;
      // Box
      auto BW = fixed_size_s * 0.5f * Fixed_ratio / IW;
      auto BH = fixed_size_s * 0.5f / Fixed_ratio / IH;
      auto C_dst_1 = op::clip(Center + (BW * BW_mask_f + BH * BH_mask_f), edsl::Tensor(0.0f), edsl::Tensor());
      auto C_dst_2 = op::clip(Center + (BW * (-BW_mask_f) + BH * (-BH_mask_f)), edsl::Tensor(), edsl::Tensor(1.0f));
      auto C_dst =
          edsl::reshape(op::concatenate({C_dst_1, C_dst_2}, -1), {H, W, fixed_ratio_count * density_s * density_s * 4});

      if (first_out) {
        C_out = C_dst;
        first_out = false;
      } else {
        C_out = op::concatenate({C_out, C_dst}, -1);
      }
    } else {
      // Density
      if (attrs.density.size()) {
        float box_width, box_height;
        box_width = box_height = attrs.fixed_size[s] * 0.5f;

        // Center
        auto CW_d =
            edsl::reshape(CW_normalized, {H, W, 1, 1, 1}) + edsl::reshape(Density_w, {1, 1, density_s, density_s, 1});
        auto CH_d =
            edsl::reshape(CH_normalized, {H, W, 1, 1, 1}) + edsl::reshape(Density_h, {1, 1, density_s, density_s, 1});
        auto Center =
            CW_d * edsl::reshape(CW_mask_f, {1, 1, 1, 1, 2}) + CH_d * edsl::reshape(CH_mask_f, {1, 1, 1, 1, 2});
        // Box
        auto BW = edsl::reshape(box_width / IW, {1, 1});
        auto BH = edsl::reshape(box_height / IH, {1, 1});
        auto C_dst_1 = op::clip(Center + (BW * BW_mask_f + BH * BH_mask_f), edsl::Tensor(0.0f), edsl::Tensor());
        auto C_dst_2 = op::clip(Center + (BW * (-BW_mask_f) + BH * (-BH_mask_f)), edsl::Tensor(), edsl::Tensor(1.0f));
        auto C_dst = edsl::reshape(op::concatenate({C_dst_1, C_dst_2}, -1), {H, W, density_s * density_s * 4});

        if (first_out) {
          C_out = C_dst;
          first_out = false;
        } else {
          C_out = op::concatenate({C_out, C_dst}, -1);
        }
      }

      // Aspect_ratio
      int ar_sz_count = aspect_ratios_scale_size.size();
      if (ar_sz_count > 0) {
        TensorShape shape_ar_sz(precision, {1, 1, ar_sz_count, 1, 1, 1});
        Buffer buffer_ar_sz(shape_ar_sz);
        buffer_ar_sz.copy_from(aspect_ratios_scale_size.data());
        auto Aspect_ratio_sz = edsl::Constant(buffer_ar_sz, "aspect_ratio_scale_size");

        // Center
        auto CW_d = edsl::reshape(CW_normalized, {H, W, 1, 1, 1, 1}) + Density_w;
        auto CH_d = edsl::reshape(CH_normalized, {H, W, 1, 1, 1, 1}) + Density_h;
        auto Center = CW_d * CW_mask_f + CH_d * CH_mask_f;

        // Box
        auto BW = fixed_size_s * 0.5f * Aspect_ratio_sz / IW;
        auto BH = fixed_size_s * 0.5f / Aspect_ratio_sz / IH;

        auto C_dst_1 = op::clip(Center + (BW * BW_mask_f + BH * BH_mask_f), edsl::Tensor(0.0f), edsl::Tensor());
        auto C_dst_2 = op::clip(Center + (BW * (-BW_mask_f) + BH * (-BH_mask_f)), edsl::Tensor(), edsl::Tensor(1.0f));
        auto C_dst =
            edsl::reshape(op::concatenate({C_dst_1, C_dst_2}, -1), {H, W, ar_sz_count * density_s * density_s * 4});

        if (first_out) {
          C_out = C_dst;
          first_out = false;
        } else {
          C_out = op::concatenate({C_out, C_dst}, -1);
        }
      }
    }
  }
}

void PriorBoxImpl::processMinSizePath() {
  // Center
  auto Center = CW_normalized * CW_mask + CH_normalized * CH_mask;
  auto BW = Min_size * 0.5 / IW;
  auto BH = Min_size * 0.5 / IH;
  auto C_min_size = Center + (BW * BW_mask + BH * BH_mask);
  C_out = C_min_size;

  // Max_size
  edsl::Tensor C_max_size;
  int max_element_size = attrs.max_size.size();
  int result_size = max_element_size < min_element_size ? max_element_size : min_element_size;
  if (attrs.scale_all_sizes && max_element_size > 0) {
    std::vector<float> max_size_new(attrs.max_size.begin(), attrs.max_size.begin() + result_size);
    TensorShape shape_max(precision, {1, 1, result_size, 1});
    Buffer buffer_max(shape_max);
    buffer_max.copy_from(max_size_new.data());
    auto Max_size = edsl::Constant(buffer_max, "buffer_max");
    auto E =
        edsl::sqrt(op::slice(Min_size).add_dim(0, 1).add_dim(0, 1).add_dim(0, result_size).add_dim(0, 1) * Max_size) *
        0.5f;
    auto BW = E / IW;
    auto BH = E / IH;
    C_max_size = Center + (BW * BW_mask + BH * BH_mask);
    // In this case, process index which min_size and max_size both have
    if (min_element_size <= max_element_size) {
      C_out = op::concatenate({C_min_size, C_max_size}, -1);
    } else {
      C_out = op::concatenate(
          {edsl::reshape(
               op::concatenate({op::slice(C_min_size).add_dim(0, H).add_dim(0, W).add_dim(0, result_size).add_dim(0, 4),
                                C_max_size},
                               -1),
               {H, W, result_size * 8}),
           edsl::reshape(
               op::slice(C_min_size).add_dim(0, H).add_dim(0, W).add_dim(result_size, min_element_size).add_dim(0, 4),
               {H, W, (min_element_size - result_size) * 4})},
          -1);
    }
  }

  // Aspect_ratio
  if (aspect_ratios_scale_size.size() > 0) {
    int ar_size = aspect_ratios_scale_size.size();
    TensorShape shape_arss(precision, {1, 1, 1, ar_size, 1});
    Buffer buffer_arss(shape_arss);
    buffer_arss.copy_from(aspect_ratios_scale_size.data());
    auto Arss = edsl::Constant(buffer_arss, "aspect_ratios_scale_size");

    edsl::Tensor Min_size_ar = attrs.scale_all_sizes ? Min_size : op::slice(Min_size).add_dims({0, 0, 0, 0});
    int msa_size = attrs.scale_all_sizes ? min_element_size : 1;
    auto Min_size_ar_ex = edsl::reshape(Min_size_ar, {1, 1, msa_size, 1, 1});
    auto BW = Min_size_ar_ex * 0.5f * Arss / IW;
    auto BH = Min_size_ar_ex * 0.5f / Arss / IH;
    auto C_ar = edsl::reshape(Center, {H, W, 1, 1, 4}) + (BW * BW_mask + BH * BH_mask);
    if (attrs.scale_all_sizes) {
      auto C_ar_reshape = edsl::reshape(C_ar, {H, W, min_element_size, 4 * ar_size});
      if (min_element_size <= max_element_size) {
        C_out = op::concatenate({C_min_size, C_max_size, C_ar_reshape}, -1);
      } else {
        // The first part will work with min_size, max_size, aspect_ratio
        // Then the left parts will work without max_size
        C_out = op::concatenate(
            {edsl::reshape(op::concatenate(
                               {op::slice(C_min_size).add_dim(0, H).add_dim(0, W).add_dim(0, result_size).add_dim(0, 4),
                                C_max_size,
                                op::slice(C_ar_reshape)
                                    .add_dim(0, H)
                                    .add_dim(0, W)
                                    .add_dim(0, result_size)
                                    .add_dim(0, 4 * ar_size)},
                               -1),
                           {H, W, result_size * (8 + 4 * ar_size)}),
             edsl::reshape(op::concatenate({op::slice(C_min_size)
                                                .add_dim(0, H)
                                                .add_dim(0, W)
                                                .add_dim(result_size, min_element_size)
                                                .add_dim(0, 4),
                                            op::slice(C_ar_reshape)
                                                .add_dim(0, H)
                                                .add_dim(0, W)
                                                .add_dim(result_size, min_element_size)
                                                .add_dim(0, 4 * ar_size)},
                                           -1),
                           {H, W, (min_element_size - result_size) * (4 + 4 * ar_size)})},
            -1);
      }
    } else {
      C_out = op::concatenate({C_min_size, edsl::reshape(C_ar, {H, W, ar_size, 4})}, -2);
    }
  }
}

void PriorBoxImpl::processVariance() {
  int channel_size = W * H * num_of_prior * 4;
  auto C_out_reshape = edsl::reshape(C_out, {1, channel_size});
  if (attrs.clip) C_out_reshape = op::clip(C_out_reshape, edsl::Tensor(0.0f), edsl::Tensor(1.0f));

  edsl::Tensor Varian;
  if (variance.size() == 1) {
    auto Val = edsl::cast(edsl::Tensor(variance[0]), precision);
    Varian = op::broadcast(Val, {1, channel_size}, {});
  } else {
    TensorShape shape_va(precision, {1, 4});
    Buffer buffer_va(shape_va);
    buffer_va.copy_from(variance.data());
    auto V = edsl::Constant(buffer_va, "variance");
    Varian = edsl::reshape(op::repeat(V).count(channel_size / 4).axis(0), {1, channel_size});
  }

  C_out = op::concatenate({C_out_reshape, Varian}, 0);
}

void registerPriorBox() {
  registerOp("PriorBox", [](const Context& ctx) {
    auto* layer = ngraph::as_type<PriorBox>(ctx.layer);
    IE_ASSERT(layer);
    IE_ASSERT(ctx.operands.size() == 2);
    const ngraph::op::PriorBoxAttrs& attrs = layer->get_attrs();
    IE_ASSERT(attrs.variance.size() == 1 || attrs.variance.size() == 4 || attrs.variance.empty());
    PriorBoxImpl pbi(ctx, attrs);
    int fixed_size_count = attrs.fixed_size.size();
    if (fixed_size_count) {
      pbi.processFixedSizePath();
    } else {
      pbi.processMinSizePath();
    }
    pbi.processVariance();
    return edsl::make_tuple(pbi.getOutput());
  });
}

}  // namespace PlaidMLPlugin
