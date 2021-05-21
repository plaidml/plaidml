// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset4.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;  // NOLINT[build/namespaces]
using ngraph::opset4::Proposal;

namespace {
edsl::Tensor generate_anchors(const ngraph::op::ProposalAttrs& attrs,  //
                              const int64_t num_anchors,               //
                              const float coordinates_offset,          //
                              DType dtype) {
  int num_ratios = attrs.ratio.size();
  int num_scales = attrs.scale.size();
  IE_ASSERT(num_anchors == num_ratios * num_scales);

  auto base_size = attrs.base_size;
  float base_area = base_size * base_size;
  float center = 0.5f * (base_size - coordinates_offset);
  auto round_ratios = attrs.framework != "tensorflow";
  auto shift_anchors = attrs.framework == "tensorflow";

  // create anchors tensor from ratios and scales based on base size, in which
  // ratio = base_height / base_width where base_height * base_width = base_size * base_size
  auto ratio_buff = Buffer(TensorShape(DType::FLOAT32, {num_ratios, 1}));
  ratio_buff.copy_from(attrs.ratio.data());
  auto ratios = edsl::Constant(ratio_buff, "ratio_node");
  ratios = edsl::reshape(op::tile(ratios, {1, num_scales}), {num_anchors, 1});

  edsl::Tensor base_width, base_height;
  if (round_ratios) {
    base_width = edsl::round(edsl::sqrt(edsl::Tensor(base_area) / ratios));
    base_height = edsl::round(base_width * ratios);
  } else {
    base_width = edsl::sqrt(edsl::Tensor(base_area) / ratios);
    base_height = base_width * ratios;
  }

  auto scale_buff = Buffer(TensorShape(DType::FLOAT32, {num_scales, 1}));
  scale_buff.copy_from(attrs.scale.data());
  auto scales = edsl::Constant(scale_buff, "scale_node");
  scales = edsl::reshape(op::tile(scales, {num_ratios, 1}), {num_anchors, 1});
  auto offset_w = (base_width * scales - coordinates_offset) * 0.5f;
  auto offset_h = (base_height * scales - coordinates_offset) * 0.5f;

  auto anchor_offsets = op::concatenate({-offset_w, -offset_h, offset_w, offset_h}, 1);
  auto anchors = anchor_offsets + center;
  if (shift_anchors) {
    anchors = anchors - base_size * 0.5f;
  }
  anchors = edsl::cast(anchors, dtype);
  return anchors;
}

edsl::Tensor enumerate_proposals(edsl::Tensor class_probs,                //
                                 edsl::Tensor class_logits,               //
                                 edsl::Tensor anchors,                    //
                                 const ngraph::op::ProposalAttrs& attrs,  //
                                 const float img_H,                       //
                                 const float img_W,                       //
                                 const float scale_H,                     //
                                 const float scale_W,                     //
                                 const float coordinates_offset,          //
                                 DType dtype) {
  auto min_box_H = attrs.min_size * scale_H;
  auto min_box_W = attrs.min_size * scale_W;

  auto class_probs_shape = class_probs.compute_shape().sizes();
  auto num_batches = class_probs_shape[0];
  auto num_anchors = class_probs_shape[1] / 2;
  auto feat_H = class_probs_shape[2];
  auto feat_W = class_probs_shape[3];
  auto num_proposals = feat_H * feat_W * num_anchors;

  bool initial_clip = attrs.framework == "tensorflow";
  bool swap_xy = attrs.framework == "tensorflow";

  // used for gather index
  edsl::Tensor idx_zero = edsl::index({edsl::TensorDim(1)}, 0);
  edsl::Tensor idx_one = idx_zero + 1;
  edsl::Tensor idx_two = idx_zero + 2;
  edsl::Tensor idx_three = idx_zero + 3;

  // anchors: [num_anchors, 4] -> [feat_H * feat_W * num_anchors, 4] = [num_proposals, 4])
  anchors = op::tile(anchors, {static_cast<int>(feat_H * feat_W)});
  edsl::Tensor anchor_wm = edsl::gather(anchors, idx_zero).axis(1);
  edsl::Tensor anchor_hm = edsl::gather(anchors, idx_one).axis(1);
  edsl::Tensor anchor_wp = edsl::gather(anchors, idx_two).axis(1);
  edsl::Tensor anchor_hp = edsl::gather(anchors, idx_three).axis(1);

  auto dim_h = edsl::TensorDim(feat_H);
  auto dim_w = edsl::TensorDim(feat_W);
  auto dim_anchor = edsl::TensorDim(num_anchors);
  auto feat_h = edsl::reshape(edsl::index({dim_h, dim_w, dim_anchor}, 0), {num_proposals, 1});
  auto feat_w = edsl::reshape(edsl::index({dim_h, dim_w, dim_anchor}, 1), {num_proposals, 1});
  auto img_x = edsl::cast((swap_xy ? feat_h : feat_w) * attrs.feat_stride, dtype);
  auto img_y = edsl::cast((swap_xy ? feat_w : feat_h) * attrs.feat_stride, dtype);

  // anchor_logits: [num_batches, num_anchors * 4, feat_H, feat_W] -> [num_batches, feat_H * feat_W * num_anchors, 4]
  auto anchor_logits = edsl::reshape(class_logits, {num_batches, num_anchors, 4, feat_H, feat_W});
  anchor_logits = op::transpose(anchor_logits, edsl::make_tuple<int64_t>({0, 3, 4, 1, 2}));
  anchor_logits = edsl::reshape(anchor_logits, {num_batches, num_proposals, 4});

  edsl::Tensor dx = edsl::gather(anchor_logits, idx_zero).axis(2);
  edsl::Tensor dy = edsl::gather(anchor_logits, idx_one).axis(2);
  edsl::Tensor d_log_w = edsl::gather(anchor_logits, idx_two).axis(2);
  edsl::Tensor d_log_h = edsl::gather(anchor_logits, idx_three).axis(2);
  dx = dx / attrs.box_coordinate_scale;
  dy = dy / attrs.box_coordinate_scale;
  d_log_w = d_log_w / attrs.box_size_scale;
  d_log_h = d_log_h / attrs.box_size_scale;

  // box upper-left corner location
  auto box_x0 = img_x + anchor_wm;
  auto box_y0 = img_y + anchor_hm;
  // box lower-right corner location
  auto box_x1 = img_x + anchor_wp;
  auto box_y1 = img_y + anchor_hp;

  if (initial_clip) {
    box_x0 = op::clip(box_x0, edsl::Tensor(0), edsl::Tensor(img_W));
    box_y0 = op::clip(box_y0, edsl::Tensor(0), edsl::Tensor(img_H));
    box_x1 = op::clip(box_x1, edsl::Tensor(0), edsl::Tensor(img_W));
    box_y1 = op::clip(box_y1, edsl::Tensor(0), edsl::Tensor(img_H));
  }

  auto box_width = box_x1 - box_x0 + coordinates_offset;
  auto box_height = box_y1 - box_y0 + coordinates_offset;
  auto box_ctr_x = box_x0 + box_width * 0.5f;
  auto box_ctr_y = box_y0 + box_height * 0.5f;

  auto pred_ctr_x = box_ctr_x + dx * box_width;
  auto pred_ctr_y = box_ctr_y + dy * box_height;
  auto pred_width = edsl::exp(d_log_w) * box_width;
  auto pred_height = edsl::exp(d_log_h) * box_height;

  box_x0 = pred_ctr_x - pred_width * 0.5f;
  box_y0 = pred_ctr_y - pred_height * 0.5f;
  box_x1 = pred_ctr_x + pred_width * 0.5f;
  box_y1 = pred_ctr_y + pred_height * 0.5f;

  if (attrs.clip_before_nms) {
    box_x0 = op::clip(box_x0, edsl::Tensor(0), edsl::Tensor(img_W - coordinates_offset));
    box_y0 = op::clip(box_y0, edsl::Tensor(0), edsl::Tensor(img_H - coordinates_offset));
    box_x1 = op::clip(box_x1, edsl::Tensor(0), edsl::Tensor(img_W - coordinates_offset));
    box_y1 = op::clip(box_y1, edsl::Tensor(0), edsl::Tensor(img_H - coordinates_offset));
  }

  auto new_box_width = box_x1 - box_x0 + coordinates_offset;
  auto new_box_height = box_y1 - box_y0 + coordinates_offset;

  // anchor_score: [num_batches, 2 * num_anchors, feat_H, feat_W] -> [num_batches, feat_H * feat_W * num_anchors, 2]
  auto anchor_score = edsl::reshape(class_probs, {num_batches, 2, num_anchors, feat_H, feat_W});
  anchor_score = op::transpose(anchor_score, edsl::make_tuple<int64_t>({0, 3, 4, 2, 1}));
  anchor_score = edsl::reshape(anchor_score, {num_batches, num_proposals, 2});
  // Currently only takes backend scores referring to openvino implementation
  anchor_score = edsl::gather(anchor_score, idx_one).axis(2);
  auto valid_box_size = (new_box_width >= min_box_W) * (new_box_height >= min_box_H);
  auto zero = edsl::cast(edsl::Tensor(0), anchor_score.dtype());
  auto proposal_score = edsl::select(valid_box_size, anchor_score, zero);

  // proposals: [num_batches, num_proposals, 5]
  auto proposals = op::concatenate({box_x0, box_y0, box_x1, box_y1, proposal_score}, 2);
  return proposals;
}

edsl::Tensor partial_sort(edsl::Tensor proposals, int64_t pre_nms_topn, int64_t num_batches) {
  edsl::Tensor idx_zero = edsl::index({edsl::TensorDim(1)}, 0);

  edsl::Tensor scores = edsl::gather(proposals, idx_zero + 4).axis(2);

  // pick first pre_nms_topn proposals
  auto topk_result = op::topk(scores, pre_nms_topn)
                         .axis(1)
                         .sort_direction(edsl::SortDirection::DESC)
                         .sort_type(op::TopKSortType::VALUE)
                         .build();
  auto topn_indices = topk_result[1];
  edsl::Tensor sorted_proposals = edsl::gather(proposals, topn_indices).mode(edsl::GatherMode::ND).batchDims(1);
  return sorted_proposals;
}

std::vector<edsl::Tensor> retrieve_rois(edsl::Tensor proposals,                  //
                                        edsl::Tensor roi_indices,                //
                                        const ngraph::op::ProposalAttrs& attrs,  //
                                        const float img_H,                       //
                                        const float img_W,                       //
                                        const int64_t num_batches) {
  edsl::Tensor idx_zero = edsl::index({edsl::TensorDim(1)}, 0);
  auto zero = edsl::cast(idx_zero, roi_indices.dtype());

  // Process nms output roi indices, make it only address 0 coordinates when index is -1
  edsl::Tensor selected_batch_indices = edsl::gather(roi_indices, idx_zero).axis(1);
  auto batch_indices = edsl::select(selected_batch_indices > -1, selected_batch_indices, zero);
  edsl::Tensor selected_box_indices = edsl::gather(roi_indices, idx_zero + 2).axis(1);
  auto box_indices = edsl::select(selected_box_indices > -1, selected_box_indices, zero + attrs.pre_nms_topn);
  roi_indices = op::concatenate({batch_indices, box_indices}, 1);

  auto zero_proposal = op::broadcast(zero, {num_batches, 1, 5}, {0});
  proposals = op::concatenate({proposals, zero_proposal}, 1);

  // selected_rois: [num_batch, post_nms_topn, 5]
  edsl::Tensor selected_rois = edsl::gather(proposals, roi_indices).mode(edsl::GatherMode::ND);

  edsl::Tensor x0 = edsl::gather(selected_rois, idx_zero).axis(1);
  edsl::Tensor y0 = edsl::gather(selected_rois, idx_zero + 1).axis(1);
  edsl::Tensor x1 = edsl::gather(selected_rois, idx_zero + 2).axis(1);
  edsl::Tensor y1 = edsl::gather(selected_rois, idx_zero + 3).axis(1);
  edsl::Tensor score = edsl::gather(selected_rois, idx_zero + 4).axis(1);

  if (attrs.clip_after_nms) {
    x0 = op::clip(x0, edsl::Tensor(0), edsl::Tensor(img_W));
    y0 = op::clip(y0, edsl::Tensor(0), edsl::Tensor(img_H));
    x1 = op::clip(x1, edsl::Tensor(0), edsl::Tensor(img_W));
    y1 = op::clip(y1, edsl::Tensor(0), edsl::Tensor(img_H));
  }
  if (attrs.normalize) {
    x0 = x0 / img_W;
    y0 = y0 / img_H;
    x1 = x1 / img_W;
    y1 = y1 / img_H;
  }

  // Only keep first -1 in batch indices, set the remaining -1 to be 0
  // -1 is the flag of termination referring to OV implementation`1
  auto item_flag = edsl::select(selected_batch_indices > -1,
                                edsl::cast(edsl::Tensor(0), selected_batch_indices.dtype()), selected_batch_indices);
  item_flag = op::cumsum(item_flag, 0);
  item_flag = edsl::select(item_flag < -1, zero, zero + 1);
  auto item_idx = selected_batch_indices * item_flag;
  item_idx = edsl::cast(item_idx, selected_rois.dtype());

  auto result_rois = op::concatenate({item_idx, x0, y0, x1, y1}, 1);
  if (attrs.infer_probs) {
    score = op::squeeze(score, {1});
    return {result_rois, score};
  } else {
    return {result_rois};
  }
}
}  // namespace

namespace PlaidMLPlugin {

void registerProposal() {
  registerOp("Proposal", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset4::Proposal>(ctx.layer);
    IE_ASSERT(ctx.operands.size() == 3);
    auto class_probs = ctx.operands.at(0);
    auto class_logits = ctx.operands.at(1);
    auto image_info = get_axis_vector_from_constant_operand(2, layer);
    IE_ASSERT(image_info.size() == 3 || image_info.size() == 4);
    auto proposal_attrs = layer->get_attrs();

    auto img_H = image_info.at(0);
    auto img_W = image_info.at(1);
    auto scale_H = image_info.at(2);
    auto scale_W = image_info.size() < 4 ? scale_H : image_info.at(3);

    auto class_probs_shape = class_probs.compute_shape().sizes();
    auto num_batches = class_probs_shape[0];
    auto num_anchors = class_probs_shape[1] / 2;
    auto feat_H = class_probs_shape[2];
    auto feat_W = class_probs_shape[3];
    auto num_proposals = num_anchors * feat_H * feat_W;

    auto pre_nms_topn = num_proposals < proposal_attrs.pre_nms_topn ? num_proposals : proposal_attrs.pre_nms_topn;
    auto post_nms_topn = proposal_attrs.post_nms_topn;
    auto coordinates_offset = proposal_attrs.framework == "tensorflow" ? 0.0f : 1.0f;

    auto anchors = generate_anchors(proposal_attrs, num_anchors, coordinates_offset, class_logits.dtype());
    auto proposals = enumerate_proposals(class_probs,         //
                                         class_logits,        //
                                         anchors,             //
                                         proposal_attrs,      //
                                         img_H,               //
                                         img_W,               //
                                         scale_H,             //
                                         scale_W,             //
                                         coordinates_offset,  //
                                         class_logits.dtype());

    // partial sort - topk
    auto sorted_proposals = partial_sort(proposals, pre_nms_topn, num_batches);

    // Prepare inputs for nms
    // boxes: [num_batches, pre_nms_topn, 4], scores: [num_batches, 1, pre_nms_topn]
    edsl::Tensor idx_zero = edsl::index({edsl::TensorDim(1)}, 0);
    edsl::Tensor box_x0 = edsl::gather(sorted_proposals, idx_zero).axis(2);
    edsl::Tensor box_y0 = edsl::gather(sorted_proposals, idx_zero + 1).axis(2);
    edsl::Tensor box_x1 = edsl::gather(sorted_proposals, idx_zero + 2).axis(2);
    edsl::Tensor box_y1 = edsl::gather(sorted_proposals, idx_zero + 3).axis(2);
    auto nms_boxes = op::concatenate({box_y0, box_x0, box_y1 + coordinates_offset, box_x1 + coordinates_offset}, 2);

    edsl::Tensor nms_scores = edsl::gather(sorted_proposals, idx_zero + 4).axis(2);
    nms_scores = edsl::reshape(nms_scores, {num_batches, 1, static_cast<int64_t>(pre_nms_topn)});
    edsl::Tensor zero = edsl::cast(edsl::Tensor(0), class_probs.dtype());
    auto iou_thresh = zero + proposal_attrs.nms_thresh;
    auto score_thresh = zero;

    // nms
    // nms_scores + 1.0f to avoid all zero scores input confuses nms
    std::vector<edsl::Tensor> nms_results =
        op::nms(nms_boxes, nms_scores + 1.0f, iou_thresh, score_thresh, post_nms_topn)
            .sort_result_descending(true)
            .hard_suppression(false)
            .build();

    // retrieve proposals
    auto result = retrieve_rois(sorted_proposals,  //
                                nms_results[0],    //
                                proposal_attrs,    //
                                img_H,             //
                                img_W,             //
                                num_batches);

    edsl::Tensor selected_box_indices = edsl::gather(nms_results[0], idx_zero + 2).axis(1);
    return edsl::make_tuple(result);
  });
}
}  // namespace PlaidMLPlugin
