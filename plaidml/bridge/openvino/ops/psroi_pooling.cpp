// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset5.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]
using namespace edsl;             // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

const static int BOX_ELEMENT_SIZE = 5;  // NOLINT

void registerPSROIPooling() {
  registerOp("PSROIPooling", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset5::PSROIPooling>(ctx.layer);
    IE_ASSERT(ctx.operands.size() == 2);
    auto I = ctx.operands.at(0);             // Input
    auto coords_boxes = ctx.operands.at(1);  // Coord
    auto coords_shapes = coords_boxes.compute_shape().sizes();

    // Get attributes about the operation.
    auto I_shape = layer->get_input_shape(0);
    auto mode = layer->get_mode();
    auto spatial_scale = layer->get_spatial_scale();
    int64_t output_dim = layer->get_output_dim();
    int64_t group_size = layer->get_group_size();
    int64_t spatial_bins_w = layer->get_spatial_bins_x();
    int64_t spatial_bins_h = layer->get_spatial_bins_y();
    int64_t channel_in = I_shape[1];
    int64_t height = I_shape[2];
    int64_t width = I_shape[3];
    int64_t num_rois = coords_shapes[0];
    int64_t num_classes = output_dim;
    int64_t pooling_h = group_size;
    int64_t pooling_w = group_size;

    if (output_dim * spatial_bins_w * spatial_bins_h != channel_in) {
      THROW_IE_EXCEPTION << "Incorrected channel of the input tensor.";
    }

    edsl::Tensor zero = edsl::index({edsl::TensorDim(1)}, 0);
    edsl::Tensor output;
    std::vector<edsl::Tensor> roi_outs;
    for (size_t roi = 0; roi < num_rois; ++roi) {
      edsl::Tensor box = edsl::gather(coords_boxes, roi);
      box = edsl::reshape(box, std::vector<int64_t>{BOX_ELEMENT_SIZE});
      auto batch_id = edsl::gather(box, 0);
      edsl::Tensor slice_I = edsl::gather(I, batch_id).axis(0);
      slice_I = op::squeeze(slice_I, {0});

      std::vector<edsl::Tensor> coords;
      for (int j = 1; j < BOX_ELEMENT_SIZE; j++) {
        coords.push_back(edsl::gather(box, j));
      }

      // Get the start and end coordinate of the box.
      auto start_w = coords[0];
      auto start_h = coords[1];
      auto end_w = coords[2];
      auto end_h = coords[3];
      if (mode == "average") {
        start_w = (edsl::round(start_w)) * spatial_scale;
        start_h = (edsl::round(start_h)) * spatial_scale;
        end_w = (edsl::round(end_w) + 1.0f) * spatial_scale;
        end_h = (edsl::round(end_h) + 1.0f) * spatial_scale;
        auto box_width = end_w - start_w;
        auto box_height = end_h - start_h;
        auto bin_width = box_width / pooling_w;
        auto bin_height = box_height / pooling_h;

        std::vector<edsl::Tensor> single_bin_outs;
        for (int64_t iph = 0; iph < pooling_h; ++iph) {
          auto ph = zero + iph;
          for (int64_t ipw = 0; ipw < pooling_w; ++ipw) {
            auto pw = zero + ipw;
            auto c_base = edsl::index({edsl::TensorDim(output_dim)}, 0);
            auto idx_c = c_base * group_size * group_size + ph * group_size + pw;
            auto bin_start_w =
                op::minimum(edsl::floor(start_w + pw * bin_width), edsl::cast(Tensor(width - 1), start_w.dtype()));
            auto bin_start_h =
                op::minimum(edsl::floor(start_h + ph * bin_height), edsl::cast(Tensor(height - 1), start_h.dtype()));
            auto bin_end_w =
                op::minimum(edsl::ceil(start_w + (pw + 1) * bin_width), edsl::cast(Tensor(width), end_w.dtype()));
            auto bin_end_h =
                op::minimum(edsl::ceil(start_h + (ph + 1) * bin_height), edsl::cast(Tensor(height), end_h.dtype()));
            auto idx_h = edsl::cast(edsl::index({TensorDim(height)}, 0), bin_start_h.dtype());
            auto idx_w = edsl::cast(edsl::index({TensorDim(width)}, 0), bin_start_w.dtype());
            idx_h = edsl::select(idx_h >= bin_start_h && idx_h <= bin_end_h - 1, idx_h,
                                 edsl::cast(Tensor(-1), idx_h.dtype()));
            idx_w = edsl::select(idx_w >= bin_start_w && idx_w <= bin_end_w - 1, idx_w,
                                 edsl::cast(Tensor(-1), idx_w.dtype()));

            auto crop_I = edsl::gather(slice_I, idx_c).axis(0);
            crop_I = edsl::gather(crop_I, idx_w)
                         .axis(2)
                         .interpolationMode(edsl::InterpolationMode::LINEAR)
                         .outOfBoundsMode(edsl::OutOfBoundsMode::RETURN_ZERO);
            crop_I = edsl::gather(crop_I, idx_h)
                         .axis(1)
                         .interpolationMode(edsl::InterpolationMode::LINEAR)
                         .outOfBoundsMode(edsl::OutOfBoundsMode::RETURN_ZERO);
            auto single_bin_out = op::sum(crop_I, edsl::make_tuple(1, 2), true);
            single_bin_out = single_bin_out / ((bin_end_h - bin_start_h) * (bin_end_w - bin_start_w));
            single_bin_outs.push_back(single_bin_out);
          }
        }
        auto roi_out = op::concatenate(single_bin_outs, 1);
        roi_out = edsl::reshape(roi_out, {1, output_dim, pooling_h, pooling_w});
        roi_outs.push_back(roi_out);
      } else if (mode == "bilinear") {
        start_w = start_w * spatial_scale;
        start_h = start_h * spatial_scale;
        end_w = end_w * spatial_scale;
        end_h = end_h * spatial_scale;
        auto box_width = end_w - start_w;
        auto box_height = end_h - start_h;
        auto bin_width = box_width / spatial_bins_w;
        auto bin_height = box_height / spatial_bins_h;
        edsl::Tensor width_scale = edsl::Tensor(0);
        edsl::Tensor height_scale = edsl::Tensor(0);
        if (pooling_w > 1) width_scale = bin_width * (width - 1) / (pooling_w - 1);
        if (pooling_h > 1) height_scale = bin_height * (height - 1) / (pooling_h - 1);
        auto c_base = edsl::cast(edsl::index({edsl::TensorDim(output_dim)}, 0), DType::FLOAT32);
        c_base = op::broadcast(c_base, {output_dim, pooling_h, pooling_w, spatial_bins_h, spatial_bins_w}, {0});

        auto ph = edsl::index(
            {TensorDim(pooling_h), TensorDim(pooling_w), TensorDim(spatial_bins_h), TensorDim(spatial_bins_w)}, 0);
        auto pw = edsl::index(
            {TensorDim(pooling_h), TensorDim(pooling_w), TensorDim(spatial_bins_h), TensorDim(spatial_bins_w)}, 1);
        auto sbh = edsl::index(
            {TensorDim(pooling_h), TensorDim(pooling_w), TensorDim(spatial_bins_h), TensorDim(spatial_bins_w)}, 2);
        auto sbw = edsl::index(
            {TensorDim(pooling_h), TensorDim(pooling_w), TensorDim(spatial_bins_h), TensorDim(spatial_bins_w)}, 3);
        ph = edsl::cast(ph, DType::FLOAT32);
        pw = edsl::cast(pw, DType::FLOAT32);
        sbh = edsl::cast(sbh, DType::FLOAT32);
        sbw = edsl::cast(sbw, DType::FLOAT32);
        auto bin_start_w = start_w + sbw * bin_width;
        auto bin_start_h = start_h + sbh * bin_height;
        auto idx_w = pooling_w > 1 ? (pw * width_scale + bin_start_w * (width - 1))
                                   : (bin_start_w + bin_start_w + bin_width) * (width - 1) / 2;
        auto idx_h = pooling_h > 1 ? (ph * height_scale + bin_start_h * (height - 1))
                                   : (bin_start_h + bin_start_h + bin_height) * (height - 1) / 2;
        auto idx_c = (sbh * spatial_bins_w + sbw) * num_classes;
        idx_c = edsl::reshape(idx_c, {1, pooling_h, pooling_w, spatial_bins_h, spatial_bins_w});
        idx_c = op::repeat(idx_c).count(output_dim).axis(0);
        idx_c = idx_c + c_base;

        idx_h = edsl::reshape(idx_h, {1, pooling_h, pooling_w, spatial_bins_h, spatial_bins_w});
        idx_w = edsl::reshape(idx_w, {1, pooling_h, pooling_w, spatial_bins_h, spatial_bins_w});
        idx_h = op::repeat(idx_h).count(output_dim).axis(0);
        idx_w = op::repeat(idx_w).count(output_dim).axis(0);

        auto total_size = output_dim * pooling_h * pooling_w * spatial_bins_h * spatial_bins_w;
        idx_c = edsl::reshape(idx_c, {total_size, 1});
        idx_h = edsl::reshape(idx_h, {total_size, 1});
        idx_w = edsl::reshape(idx_w, {total_size, 1});
        auto indices = op::concatenate({idx_c, idx_h, idx_w}, 1);

        edsl::Tensor crop_I = op::gatherND(slice_I, indices).interpolationMode(InterpolationMode::LINEAR);
        crop_I = edsl::reshape(crop_I, {1, output_dim, pooling_h, pooling_w, spatial_bins_h, spatial_bins_w});
        auto roi_out = op::sum(crop_I, edsl::make_tuple(4, 5), false);
        roi_out = roi_out / (spatial_bins_w * spatial_bins_h);
        roi_outs.push_back(roi_out);
      } else {
        THROW_IE_EXCEPTION << "Invalid PS ROI pooling mode.";
      }
    }
    output = op::concatenate(roi_outs, 0);
    return edsl::make_tuple(output);
  });
}

}  // namespace PlaidMLPlugin
