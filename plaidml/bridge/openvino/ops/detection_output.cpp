// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset4.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]
using namespace plaidml::edsl;

namespace PlaidMLPlugin {

edsl::Tensor decodeBoxes(edsl::Tensor priors, edsl::Tensor prior_variances, edsl::Tensor location,
                         const std::string& code_type, int input_height, int input_width, int batch, int num_priors,
                         bool clip_before_nms) {
  edsl::Tensor decoded_bboxes;
  if (code_type == "caffe.PriorBoxParameter.CORNER") {
    decoded_bboxes = priors / input_width + prior_variances * location;
  } else if (code_type == "caffe.PriorBoxParameter.CENTER_SIZE") {
    auto var_xmin = op::slice(prior_variances).add_dim(0, batch).add_dim(0, num_priors).add_dim(0, 1);
    auto var_ymin = op::slice(prior_variances).add_dim(0, batch).add_dim(0, num_priors).add_dim(1, 2);
    auto var_xmax = op::slice(prior_variances).add_dim(0, batch).add_dim(0, num_priors).add_dim(2, 3);
    auto var_ymax = op::slice(prior_variances).add_dim(0, batch).add_dim(0, num_priors).add_dim(3, 4);
    edsl::Tensor prior_xmin, prior_ymin, prior_xmax, prior_ymax;
    prior_xmin = op::slice(priors).add_dim(0, batch).add_dim(0, num_priors).add_dim(0, 1) / input_width;
    prior_ymin = op::slice(priors).add_dim(0, batch).add_dim(0, num_priors).add_dim(1, 2) / input_height;
    prior_xmax = op::slice(priors).add_dim(0, batch).add_dim(0, num_priors).add_dim(2, 3) / input_width;
    prior_ymax = op::slice(priors).add_dim(0, batch).add_dim(0, num_priors).add_dim(3, 4) / input_height;
    auto loc_xmin = op::slice(location).add_dim(0, batch).add_dim(0, num_priors).add_dim(0, 1);
    auto loc_ymin = op::slice(location).add_dim(0, batch).add_dim(0, num_priors).add_dim(1, 2);
    auto loc_xmax = op::slice(location).add_dim(0, batch).add_dim(0, num_priors).add_dim(2, 3);
    auto loc_ymax = op::slice(location).add_dim(0, batch).add_dim(0, num_priors).add_dim(3, 4);
    auto prior_w = prior_xmax - prior_xmin;
    auto prior_h = prior_ymax - prior_ymin;
    auto prior_center_x = (prior_xmin + prior_xmax) / 2.0f;
    auto prior_center_y = (prior_ymin + prior_ymax) / 2.0f;
    auto decoded_center_x = var_xmin * loc_xmin * prior_w + prior_center_x;
    auto decoded_center_y = var_ymin * loc_ymin * prior_h + prior_center_y;
    auto decoded_w = edsl::exp(var_xmax * loc_xmax) * prior_w;
    auto decoded_h = edsl::exp(var_ymax * loc_ymax) * prior_h;
    auto decoded_xmin = decoded_center_x - decoded_w / 2.0f;
    auto decoded_ymin = decoded_center_y - decoded_h / 2.0f;
    auto decoded_xmax = decoded_center_x + decoded_w / 2.0f;
    auto decoded_ymax = decoded_center_y + decoded_h / 2.0f;
    std::vector<edsl::Tensor> decoded_slices = {decoded_xmin, decoded_ymin, decoded_xmax, decoded_ymax};
    decoded_bboxes = op::concatenate(decoded_slices, 2);
  }
  if (clip_before_nms) {
    decoded_bboxes =
        op::clip(decoded_bboxes, cast(edsl::Tensor{0.0f}, DType::FLOAT32), cast(edsl::Tensor{1.0f}, DType::FLOAT32));
  }

  return decoded_bboxes;
}

void registerDetectionOutput() {
  registerOp("DetectionOutput", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset4::DetectionOutput>(ctx.layer);
    IE_ASSERT(ctx.operands.size() == 3 || ctx.operands.size() == 5);
    auto Location = ctx.operands.at(0);
    auto Confidence = ctx.operands.at(1);
    auto Priors = ctx.operands.at(2);

    edsl::Tensor ArmConfidence;
    edsl::Tensor ArmLocation;
    bool with_add_pred = false;
    if (ctx.operands.size() == 5) {
      ArmConfidence = ctx.operands.at(3);
      ArmLocation = ctx.operands.at(4);
      with_add_pred = true;
    }

    auto num_classes = layer->get_attrs().num_classes;
    auto background_label_id = layer->get_attrs().background_label_id;
    auto top_k = layer->get_attrs().top_k;
    auto variance_encoded_in_target = layer->get_attrs().variance_encoded_in_target;
    auto keep_top_k = layer->get_attrs().keep_top_k;
    auto code_type = layer->get_attrs().code_type;
    auto share_location = layer->get_attrs().share_location;
    auto nms_threshold = layer->get_attrs().nms_threshold;
    auto confidence_threshold = layer->get_attrs().confidence_threshold;
    auto clip_after_nms = layer->get_attrs().clip_after_nms;
    auto clip_before_nms = layer->get_attrs().clip_before_nms;
    auto decrease_label_id = layer->get_attrs().decrease_label_id;
    auto normalized = layer->get_attrs().normalized;
    auto input_height = layer->get_attrs().input_height;
    auto input_width = layer->get_attrs().input_width;
    auto objectness_score = layer->get_attrs().objectness_score;

    // TODO: support decrease_label_id, keep_top_k with multiple batches in NMS.
    IE_ASSERT(decrease_label_id == false);
    // TODO: support share_location = false which is not
    // sharing bounding boxes among different classes.
    IE_ASSERT(share_location == true);

    int prior_size = normalized ? 4 : 5;
    int prior_offset = normalized ? 0 : 1;
    int i_h = normalized ? 1 : input_height;
    int i_w = normalized ? 1 : input_width;
    auto batch = Location.compute_shape().sizes()[0];
    auto num_priors = Priors.compute_shape().sizes()[2] / prior_size;
    auto priors_shape_variance = Priors.compute_shape().sizes()[1];

    std::vector<int64_t> location_shape = {batch, num_priors, 4};
    std::vector<int64_t> confidence_shape = {batch, num_priors, num_classes};
    std::vector<int64_t> priors_shape = {batch, priors_shape_variance, num_priors, prior_size};

    edsl::Tensor location = edsl::reshape(Location, location_shape);
    edsl::Tensor confidence = edsl::reshape(Confidence, confidence_shape);

    // priors tensor consists of prior_boxes and prior_variances.
    edsl::Tensor priors = edsl::reshape(Priors, priors_shape);
    edsl::Tensor prior_boxes =
        op::slice(priors).add_dim(0, batch).add_dim(0, 1).add_dim(0, num_priors).add_dim(prior_offset, prior_size);
    edsl::Tensor prior_variances;
    if (variance_encoded_in_target) {
      // Fill prior_variances with 1s.
      auto one = cast(edsl::Tensor{1}, prior_boxes.dtype());
      auto var_shape = prior_boxes.compute_shape().sizes();
      prior_variances = op::broadcast(one, var_shape, {});
    } else {
      prior_variances = op::slice(priors).add_dim(0, batch).add_dim(1, 2).add_dim(0, num_priors).add_dim(0, prior_size);
      if (!normalized) {
        // Reshape the prior_variances tensor from {batch, 1, num_priors, 5} to {batch, 1, num_priors, 4};
        prior_variances = edsl::reshape(prior_variances, {batch, 1, num_priors * prior_size});
        // Truncate the prior_variances tensor and drop the trailing invalid data.
        prior_variances = op::slice(prior_variances).add_dim(0, batch).add_dim(0, 1).add_dim(0, num_priors * 4);
        prior_variances = edsl::reshape(prior_variances, {batch, 1, num_priors, 4});
      }
    }

    prior_variances = op::squeeze(prior_variances, {1});
    prior_boxes = op::squeeze(prior_boxes, {1});

    edsl::Tensor decoded_bboxes;
    if (with_add_pred) {
      // Update confidence if there are 5 inputs.
      Tensor IX = edsl::index({edsl::TensorDim(num_priors)}, 0);
      IX = IX * 2 + 1;
      edsl::Tensor arm_conf = edsl::gather(ArmConfidence, IX).axis(1);
      arm_conf = op::repeat(op::unsqueeze(arm_conf, {-1})).count(num_classes).axis(2);
      confidence = edsl::select(arm_conf < objectness_score, cast(edsl::Tensor{0.0f}, DType::FLOAT32), confidence);

      // Decode bounding boxes
      edsl::Tensor arm_loc = edsl::reshape(ArmLocation, location_shape);
      auto decoded_priors =
          decodeBoxes(prior_boxes, prior_variances, arm_loc, code_type, i_h, i_w, batch, num_priors, clip_before_nms);
      decoded_bboxes = decodeBoxes(decoded_priors, prior_variances, location, code_type, i_h, i_w, batch, num_priors,
                                   clip_before_nms);
    } else {
      decoded_bboxes =
          decodeBoxes(prior_boxes, prior_variances, location, code_type, i_h, i_w, batch, num_priors, clip_before_nms);
    }

    // Transpose the confidence to match the input shape of `scores` in NMS.
    // confidence -> {batch, num_boxes, num_classes}
    // scores -> {batch, num_classes, num_boxes}
    edsl::Tensor nms_conf = op::transpose(confidence, edsl::make_tuple<int64_t>({0, 2, 1}));
    // Set the scores of background class to zeros.
    if (background_label_id > -1) {
      auto zero = cast(edsl::Tensor{0.0f}, nms_conf.dtype());
      std::vector<int64_t> slice_shape = {batch, 1, num_priors};
      auto bg_slice = op::broadcast(zero, slice_shape, {});
      edsl::Tensor idxs = edsl::index({edsl::TensorDim(num_priors)}, 0);
      edsl::Tensor scatter_idx = op::slice(idxs).add_dim(background_label_id, background_label_id + 1);
      nms_conf = edsl::scatter(nms_conf, scatter_idx, bg_slice).axis(1).mode(edsl::ScatterMode::UPDATE_SLICE);
    }

    edsl::Tensor iou_threshold = cast(edsl::Tensor{nms_threshold}, DType::FLOAT32);
    edsl::Tensor score_threshold = cast(edsl::Tensor{confidence_threshold}, DType::FLOAT32);
    std::vector<edsl::Tensor> result = op::nms(decoded_bboxes, nms_conf, iou_threshold, score_threshold, top_k)
                                           .soft_nms_sigma(0.0f)
                                           .center_point_box(false)
                                           .sort_result_descending(false)
                                           .box_output_type(DType::INT32)
                                           .build();
    edsl::Tensor selected_indices = result[0];
    auto selected_indices_shape = selected_indices.compute_shape().sizes();
    edsl::Tensor selected_scores = result[1];
    edsl::Tensor valid_outputs = result[2];

    edsl::Tensor idxs = edsl::index({edsl::TensorDim(3)}, 0);
    edsl::Tensor batch_slice_idxs = op::slice(idxs).add_dim(0, 1);
    edsl::Tensor box_slice_idxs = op::slice(idxs).add_dim(2, 3);
    std::vector<edsl::Tensor> slice_idxs_vec = {batch_slice_idxs, box_slice_idxs};
    edsl::Tensor slice_idxs = cast(op::concatenate(slice_idxs_vec, 0), DType::INT32);
    edsl::Tensor gather_idxs = edsl::gather(selected_indices, slice_idxs).axis(1);
    edsl::Tensor out_boxes = edsl::gather(decoded_bboxes, gather_idxs).mode(GatherMode::ND);

    if (clip_after_nms) {
      out_boxes =
          op::clip(out_boxes, cast(edsl::Tensor{0.0f}, DType::FLOAT32), cast(edsl::Tensor{1.0f}, DType::FLOAT32));
    }

    edsl::Tensor O = op::concatenate({selected_scores, out_boxes}, 1);

    edsl::Tensor topk_results;
    if (keep_top_k[0] > -1 && selected_indices_shape[0] > keep_top_k[0]) {
      edsl::Tensor scores_slice = op::slice(selected_scores).add_dim(0, selected_indices_shape[0]).add_dim(2, 3);
      auto sorted_idxs = op::squeeze(edsl::argsort(scores_slice, 0, edsl::SortDirection::DESC), {-1});
      edsl::Tensor idxs_topk = edsl::gather(sorted_idxs, edsl::index({edsl::TensorDim(keep_top_k[0])}, 0));
      auto idxs_topk_sorted = op::sort(idxs_topk, 0, edsl::SortDirection::ASC);
      topk_results = edsl::gather(O, idxs_topk_sorted).axis(0);
    } else {
      // Pad -1 at the end the valid data.
      auto neg_one = cast(edsl::Tensor{-1}, selected_indices.dtype());
      // The output is a 7-element tuple.
      int output_tuple_size = 7;
      std::vector<int64_t> pad_shape = {1, output_tuple_size};
      auto pad_slice = op::broadcast(neg_one, pad_shape, {});
      topk_results = op::concatenate({O, pad_slice}, 0);
    }

    return edsl::make_tuple(topk_results);
  });
}

}  // namespace PlaidMLPlugin
