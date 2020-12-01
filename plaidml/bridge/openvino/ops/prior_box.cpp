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
  int H;
  int W;
  int IH;
  int IW;
  int num_of_prior;
  int min_element_size;
  float step;
  float step_x;
  float step_y;
  float offset;
  std::vector<float> aspect_ratios;
  std::vector<float> aspect_ratios_scale_size;
  std::vector<float> variance;
  std::vector<float> min_size;
  edsl::Tensor C_mix;
  edsl::Tensor C_out;
};

PriorBoxImpl::PriorBoxImpl(const Context& context, const ngraph::op::PriorBoxAttrs& attributes)
    : ctx(context), attrs(attributes) {
  prepareConfig();
  produceCenterBase();
}

// Preprocess data with some attributes
void PriorBoxImpl::prepareConfig() {
  auto* input_shape_ngraph_op = ngraph::as_type<ngraph::op::Constant>(ctx.layer->get_input_node_ptr(0));
  std::vector<int> input_shape = input_shape_ngraph_op->get_vector<int>();
  H = input_shape[0];
  W = input_shape[1];
  auto* image_shape_ngraph_op = ngraph::as_type<ngraph::op::Constant>(ctx.layer->get_input_node_ptr(1));
  std::vector<int> image_shape = image_shape_ngraph_op->get_vector<int>();
  IH = image_shape[0];
  IW = image_shape[1];

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

  step = attrs.step;
  min_size = attrs.min_size;
  if (!attrs.scale_all_sizes) {
    // mxnet-like PriorBox
    if (step == -1)
      step = 1.f * IH / H;
    else
      step *= IH;
    for (auto& size : min_size) size *= IH;
  }
  step_x = (step == 0) ? static_cast<float>(IW) / W : step;
  step_y = (step == 0) ? static_cast<float>(IH) / H : step;

  min_element_size = min_size.size();
  offset = attrs.offset;
}

void PriorBoxImpl::produceCenterBase() {
  auto CW = edsl::cast(edsl::index({edsl::TensorDim(H), edsl::TensorDim(W), edsl::TensorDim(1)}, 1), DType::FLOAT32);
  auto CH = edsl::cast(edsl::index({edsl::TensorDim(H), edsl::TensorDim(W), edsl::TensorDim(1)}, 0), DType::FLOAT32);
  edsl::Tensor CW_normalized, CH_normalized;
  if (step == 0) {
    CW_normalized = (CW + 0.5f) * step_x / IW;
    CH_normalized = (CH + 0.5f) * step_y / IH;
  } else {
    CW_normalized = (CW + offset) * step / IW;
    CH_normalized = (CH + offset) * step / IH;
  }
  C_mix = op::concatenate({CW_normalized, CH_normalized}, -1);
}

void PriorBoxImpl::processFixedSizePath() {
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
    auto Density_mix =
        edsl::reshape(op::concatenate({(edsl::cast(edsl::index({e, e, one}, 1), DType::FLOAT32) * shift + var) / IW,
                                       (edsl::cast(edsl::index({e, e, one}, 0), DType::FLOAT32) * shift + var) / IH},
                                      -1),
                      {density_s * density_s, 2});

    if (fixed_ratio_count) {
      std::vector<float> fixed_ratio_ar;
      for (auto ar : attrs.fixed_ratio) {
        fixed_ratio_ar.push_back(std::sqrt(ar));
      }
      TensorShape shape_fr(DType::FLOAT32, {fixed_ratio_count, 1});
      Buffer buffer_fr(shape_fr);
      buffer_fr.copy_from(fixed_ratio_ar.data());
      auto Fixed_ratio = edsl::Constant(buffer_fr, "fixed_ratio");

      // Expand center for output
      edsl::Tensor Center_temp =
          op::broadcast(C_mix, {H, W, fixed_ratio_count, density_s * density_s, 2}, {0, 1, 4}) + Density_mix;

      // Box
      auto Box_width_fixed_ratio_ar = fixed_size_s * 0.5f * Fixed_ratio / IW;
      auto Box_height_fixed_ratio_ar = fixed_size_s * 0.5f / Fixed_ratio / IH;
      auto Box_fixed_ratio_ar_first = op::concatenate({-Box_width_fixed_ratio_ar, -Box_height_fixed_ratio_ar}, -1);
      auto Box_fixed_ratio_ar_second = op::concatenate({Box_width_fixed_ratio_ar, Box_height_fixed_ratio_ar}, -1);

      // Combine two outputs (each process element which is WH)
      // Got a output which final dimension is 4 (WHWH)
      edsl::TensorIndex i, j, k, l, m;
      auto C_dst = edsl::reshape(
          op::concatenate({op::clip(edsl::Contraction()
                                        .outShape(H, W, fixed_ratio_count, density_s * density_s, 2)
                                        .outAccess(i, j, k, l, m)
                                        .assign(Center_temp(i, j, k, l, m) + Box_fixed_ratio_ar_first(k, m)),
                                    edsl::Tensor(0.0f), edsl::Tensor()),
                           op::clip(edsl::Contraction()
                                        .outShape(H, W, fixed_ratio_count, density_s * density_s, 2)
                                        .outAccess(i, j, k, l, m)
                                        .assign(Center_temp(i, j, k, l, m) + Box_fixed_ratio_ar_second(k, m)),
                                    edsl::Tensor(), edsl::Tensor(1.0f))},
                          -1),
          {H, W, fixed_ratio_count * density_s * density_s * 4});

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
        auto Center_temp = op::broadcast(C_mix, {H, W, density_s * density_s, 2}, {0, 1, 3}) + Density_mix;

        // Box
        std::vector<float> box_db;
        box_db.push_back(-box_width / IW);
        box_db.push_back(-box_height / IH);
        TensorShape shape_db(DType::FLOAT32, {2});
        Buffer buffer_db(shape_db);
        buffer_db.copy_from(box_db.data());
        auto Box_db = edsl::Constant(buffer_db, "box_db");

        // Combine two tensor with element size 2 (WH) to a tensor with element size 4 (WHWH)
        auto C_dst = edsl::reshape(op::concatenate({op::clip(Center_temp + Box_db, edsl::Tensor(0.0f), edsl::Tensor()),
                                                    op::clip(Center_temp - Box_db, edsl::Tensor(), edsl::Tensor(1.0f))},
                                                   -1),
                                   {H, W, density_s * density_s * 4});
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
        TensorShape shape_ar_sz(DType::FLOAT32, {ar_sz_count, 1});
        Buffer buffer_ar_sz(shape_ar_sz);
        buffer_ar_sz.copy_from(aspect_ratios_scale_size.data());
        auto Aspect_ratio_sz = edsl::Constant(buffer_ar_sz, "aspect_ratio_scale_size");
        auto Box_width_ar_sz = fixed_size_s * 0.5f * Aspect_ratio_sz / IW;
        auto Box_height_ar_sz = fixed_size_s * 0.5f / Aspect_ratio_sz / IH;

        // Center
        auto Center_temp = op::broadcast(C_mix, {H, W, ar_sz_count, density_s * density_s, 2}, {0, 1, 4}) + Density_mix;

        // Box
        auto Box_ar_sz_first = op::concatenate({-Box_width_ar_sz, -Box_height_ar_sz}, -1);
        auto Box_ar_sz_second = op::concatenate({Box_width_ar_sz, Box_height_ar_sz}, -1);

        // Each box have different clip style
        // Combine two tensor which element is WH to WHWH, the element size become 4
        edsl::TensorIndex i, j, k, l, m;
        auto C_dst =
            edsl::reshape(op::concatenate({op::clip(edsl::Contraction()
                                                        .outShape(H, W, ar_sz_count, density_s * density_s, 2)
                                                        .outAccess(i, j, k, l, m)
                                                        .assign(Center_temp(i, j, k, l, m) + Box_ar_sz_first(k, m)),
                                                    edsl::Tensor(0.0f), edsl::Tensor()),
                                           op::clip(edsl::Contraction()
                                                        .outShape(H, W, ar_sz_count, density_s * density_s, 2)
                                                        .outAccess(i, j, k, l, m)
                                                        .assign(Center_temp(i, j, k, l, m) + Box_ar_sz_second(k, m)),
                                                    edsl::Tensor(), edsl::Tensor(1.0f))},
                                          -1),
                          {H, W, ar_sz_count * density_s * density_s * 4});

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
  auto C_mix_box = op::concatenate({C_mix, C_mix}, -1);
  auto C = op::repeat(op::unsqueeze(C_mix_box, {-2})).count(min_element_size).axis(-2);

  // Min_size
  std::vector<float> box_min;
  for (auto i : min_size) {
    float box_width = i * 0.5;
    float box_height = i * 0.5;
    float first = -box_width / IW;
    float second = -box_height / IH;
    box_min.push_back(first);
    box_min.push_back(second);
    box_min.push_back(-first);
    box_min.push_back(-second);
  }
  TensorShape shape(DType::FLOAT32, {min_element_size, 4});
  Buffer buffer_min(shape);
  buffer_min.copy_from(box_min.data());
  auto Box_min = edsl::Constant(buffer_min, "box_min");

  auto C_min_size = C + Box_min;
  C_out = C_min_size;

  // Max_size
  edsl::Tensor C_max_size;
  int max_element_size = attrs.max_size.size();
  int result_size = max_element_size < min_element_size ? max_element_size : min_element_size;
  if (attrs.scale_all_sizes && max_element_size > 0) {
    std::vector<float> box_max;
    for (size_t i = 0; i < result_size; i++) {
      float box_width = std::sqrt(min_size[i] * attrs.max_size[i]) * 0.5f;
      float box_height = box_width;
      float first = -box_width / IW;
      float second = -box_height / IH;
      box_max.push_back(first);
      box_max.push_back(second);
      box_max.push_back(-first);
      box_max.push_back(-second);
    }
    TensorShape shape_max(DType::FLOAT32, {result_size, 4});
    Buffer buffer_max(shape_max);
    buffer_max.copy_from(box_max.data());
    auto B_max = edsl::Constant(buffer_max, "box_max");

    C_max_size = op::repeat(op::unsqueeze(C_mix_box, {-2})).count(result_size).axis(-2) + B_max;

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
    std::vector<float> min_size_ar;
    if (attrs.scale_all_sizes) {
      min_size_ar = min_size;
    } else {
      min_size_ar.push_back(min_size[0]);
    }
    std::vector<float> box_ar;
    for (auto ms : min_size_ar) {
      for (auto ar : aspect_ratios_scale_size) {
        float box_width = ms * 0.5f * ar;
        float box_height = ms * 0.5f / ar;
        float first = -box_width / IW;
        float second = -box_height / IH;
        box_ar.push_back(first);
        box_ar.push_back(second);
        box_ar.push_back(-first);
        box_ar.push_back(-second);
      }
    }
    int msa_size = min_size_ar.size();
    int ar_size = aspect_ratios_scale_size.size();
    TensorShape shape(DType::FLOAT32, {msa_size * ar_size, 4});
    Buffer buffer_ar(shape);
    buffer_ar.copy_from(box_ar.data());
    auto B_ar = edsl::Constant(buffer_ar, "box_ar");

    auto C_ar = op::repeat(op::unsqueeze(C_mix_box, {-2})).count(msa_size * ar_size).axis(-2) + B_ar;
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
    auto Val = edsl::cast(edsl::Tensor(variance[0]), DType::FLOAT32);
    Varian = op::broadcast(Val, {1, channel_size}, {});
  } else {
    TensorShape shape_va(DType::FLOAT32, {1, 4});
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
