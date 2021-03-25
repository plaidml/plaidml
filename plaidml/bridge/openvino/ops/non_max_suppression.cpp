// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset5.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;  // NOLINT[build/namespaces]
using ngraph::opset5::NonMaxSuppression;

namespace PlaidMLPlugin {

class nms {
 public:
  explicit nms(edsl::Tensor BOXES, edsl::Tensor SCORES, int32_t max_output_boxes_per_class, edsl::Tensor IOU_THRESHOLD,
               edsl::Tensor SCORE_THRESHOLD, edsl::Tensor SOFT_NMS_SIGMA)
      : BOXES_(BOXES),
        SCORES_(SCORES),
        max_output_boxes_per_class_(max_output_boxes_per_class),
        IOU_THRESHOLD_(IOU_THRESHOLD),
        SCORE_THRESHOLD_(SCORE_THRESHOLD),
        SOFT_NMS_SIGMA_(SOFT_NMS_SIGMA),
        center_point_box_(false),
        sort_result_descending_(false),
        box_input_type_(DType::FLOAT32),
        box_output_type_(DType::INT32),
        thres_type_(DType::FLOAT32) {}

  nms& center_point_box(bool center_point_box) {
    center_point_box_ = center_point_box;
    return *this;
  }

  nms& sort_result_descending(bool sort_result_descending) {
    sort_result_descending_ = sort_result_descending;
    return *this;
  }

  nms& box_input_type(DType box_input_type) {
    box_input_type_ = box_input_type;
    return *this;
  }

  nms& box_output_type(DType box_output_type) {
    box_output_type_ = box_output_type;
    return *this;
  }

  nms& thres_type(DType thres_type) {
    thres_type_ = thres_type_;
    return *this;
  }

  std::vector<edsl::Tensor> build();

 private:
  edsl::Tensor BOXES_;
  edsl::Tensor SCORES_;
  int32_t max_output_boxes_per_class_;
  edsl::Tensor IOU_THRESHOLD_;
  edsl::Tensor SCORE_THRESHOLD_;
  edsl::Tensor SOFT_NMS_SIGMA_;
  bool center_point_box_;
  bool sort_result_descending_;
  DType box_input_type_;
  DType box_output_type_;
  DType thres_type_;
};

std::vector<edsl::Tensor> nms::build() {
  std::vector<int64_t> boxes_shape = BOXES_.compute_shape().sizes();
  std::vector<int64_t> scores_shape = SCORES_.compute_shape().sizes();
  int num_batches = boxes_shape[0];
  int num_boxes = boxes_shape[1];
  int num_classes = scores_shape[1];

  // Used for gather and select
  edsl::Tensor ZERO = edsl::cast(edsl::index({edsl::TensorDim(1)}, 0), box_input_type_);
  edsl::Tensor ZERO_INT = edsl::cast(ZERO, DType::INT32);
  edsl::Tensor ONE = ZERO + 1;
  edsl::Tensor ONE_INT = ZERO_INT + 1;
  edsl::Tensor NEG1 = -ONE;
  edsl::Tensor NEG1_O = edsl::cast(NEG1, box_output_type_);

  std::vector<edsl::Tensor> boxes;
  std::vector<edsl::Tensor> scores;
  edsl::Tensor VALID_OUTPUTS = ZERO;

  edsl::Tensor BOXES_Y1;
  edsl::Tensor BOXES_X1;
  edsl::Tensor BOXES_Y2;
  edsl::Tensor BOXES_X2;
  if (center_point_box_) {
    edsl::Tensor BOXES_XCENTER = edsl::gather(BOXES_, ZERO_INT).axis(2);
    edsl::Tensor BOXES_YCENTER = edsl::gather(BOXES_, ONE_INT).axis(2);
    edsl::Tensor BOXES_WIDTH_HALF = edsl::gather(BOXES_, ONE_INT + 1).axis(2);
    BOXES_WIDTH_HALF = BOXES_WIDTH_HALF / 2.0f;
    edsl::Tensor BOXES_HEIGHT_HALF = edsl::gather(BOXES_, ONE_INT + 2).axis(2);
    BOXES_HEIGHT_HALF = BOXES_HEIGHT_HALF / 2.0f;
    BOXES_X1 = BOXES_XCENTER - BOXES_WIDTH_HALF;
    BOXES_X2 = BOXES_XCENTER + BOXES_WIDTH_HALF;
    BOXES_Y1 = BOXES_YCENTER - BOXES_HEIGHT_HALF;
    BOXES_Y2 = BOXES_YCENTER + BOXES_HEIGHT_HALF;
  } else {
    BOXES_Y1 = edsl::gather(BOXES_, ZERO_INT).axis(2);
    BOXES_X1 = edsl::gather(BOXES_, ONE_INT).axis(2);
    BOXES_Y2 = edsl::gather(BOXES_, ONE_INT + 1).axis(2);
    BOXES_X2 = edsl::gather(BOXES_, ONE_INT + 2).axis(2);
  }

  BOXES_Y1 = edsl::reshape(BOXES_Y1, {num_batches, num_boxes});
  BOXES_X1 = edsl::reshape(BOXES_X1, {num_batches, num_boxes});
  BOXES_Y2 = edsl::reshape(BOXES_Y2, {num_batches, num_boxes});
  BOXES_X2 = edsl::reshape(BOXES_X2, {num_batches, num_boxes});

  edsl::Tensor IOU_AREAI = (BOXES_Y2 - BOXES_Y1) * (BOXES_X2 - BOXES_X1);
  edsl::Tensor IOU_INTERSECTION_YMIN = op::maximum(edsl::reshape(BOXES_Y1, {num_batches, num_boxes, 1}),
                                                   edsl::reshape(BOXES_Y1, {num_batches, 1, num_boxes}));
  edsl::Tensor IOU_INTERSECTION_XMIN = op::maximum(edsl::reshape(BOXES_X1, {num_batches, num_boxes, 1}),
                                                   edsl::reshape(BOXES_X1, {num_batches, 1, num_boxes}));
  edsl::Tensor IOU_INTERSECTION_YMAX = op::minimum(edsl::reshape(BOXES_Y2, {num_batches, num_boxes, 1}),
                                                   edsl::reshape(BOXES_Y2, {num_batches, 1, num_boxes}));
  edsl::Tensor IOU_INTERSECTION_XMAX = op::minimum(edsl::reshape(BOXES_X2, {num_batches, num_boxes, 1}),
                                                   edsl::reshape(BOXES_X2, {num_batches, 1, num_boxes}));
  edsl::Tensor IOU_INTERSECTION_AREA_YGAP = IOU_INTERSECTION_YMAX - IOU_INTERSECTION_YMIN;
  edsl::Tensor IOU_INTERSECTION_AREA_XGAP = IOU_INTERSECTION_XMAX - IOU_INTERSECTION_XMIN;
  edsl::Tensor IOU_INTERSECTION_AREA =
      edsl::select(IOU_INTERSECTION_AREA_YGAP > 0.0f, IOU_INTERSECTION_AREA_YGAP, ZERO) *
      edsl::select(IOU_INTERSECTION_AREA_XGAP > 0.0f, IOU_INTERSECTION_AREA_XGAP, ZERO);

  edsl::Tensor IOU_DENOMINATOR =
      op::unsqueeze(IOU_AREAI, {-1}) + op::unsqueeze(IOU_AREAI, {-2}) - IOU_INTERSECTION_AREA;

  edsl::Tensor IOU_DENOMINATOR_ZEROED = edsl::select(IOU_DENOMINATOR <= 0.0f, ZERO, 1.0f / IOU_DENOMINATOR);
  edsl::Tensor IOU = IOU_INTERSECTION_AREA * IOU_DENOMINATOR_ZEROED;

  edsl::Tensor WEIGHT = edsl::cast(
      edsl::select(SOFT_NMS_SIGMA_ != 0.0f, -0.5f / SOFT_NMS_SIGMA_, edsl::cast(ZERO, thres_type_)), box_input_type_);

  TensorShape NODE_SHAPE(DType::FLOAT32, {1, 1, 2});

  std::vector<float> invalid_node_index = {-1, -1};
  Buffer buffer_invalid_node(NODE_SHAPE);
  buffer_invalid_node.copy_from(invalid_node_index.data());
  auto INVALID_NODE = edsl::cast(edsl::Constant(buffer_invalid_node, "INVALID_NODE"), box_output_type_);

  int num_boxes_per_class = std::min(num_boxes, max_output_boxes_per_class_);

  auto INDEX_B = edsl::index({edsl::TensorDim(num_batches), edsl::TensorDim(num_classes), edsl::TensorDim(1)}, 0);
  auto INDEX_C = edsl::index({edsl::TensorDim(num_batches), edsl::TensorDim(num_classes), edsl::TensorDim(1)}, 1);
  auto INDEX_BC = edsl::cast(op::concatenate({INDEX_B, INDEX_C}, 2), box_output_type_);

  edsl::Tensor SCORE_THRESHOLD_I = edsl::cast(SCORE_THRESHOLD_, box_input_type_);
  edsl::Tensor IOU_THRESHOLD_I = edsl::cast(IOU_THRESHOLD_, box_input_type_);
  edsl::Tensor ZERO_SCATTER = op::broadcast(ZERO, {num_batches, num_classes, 1}, {0});
  edsl::Tensor NEW_SCORES = edsl::select(SCORES_ > SCORE_THRESHOLD_I, SCORES_, ZERO);

  // Select box
  for (int k = 0; k < num_boxes_per_class; k++) {
    // Select the box with largest score
    edsl::Tensor CANDIDATE_INDEX =
        edsl::gather(edsl::argsort(NEW_SCORES, 2, edsl::SortDirection::DESC), ZERO_INT).axis(2);
    edsl::Tensor SCORE = edsl::reshape(op::max(NEW_SCORES, edsl::Value(2)), {num_batches, num_classes, 1});
    edsl::Tensor CURRENT_NODE = edsl::select(SCORE > 0.0f, INDEX_BC, INVALID_NODE);

    // Update count of selected box
    edsl::Tensor VALID = op::sum(edsl::select(SCORE != 0.0f, ONE, ZERO));
    VALID_OUTPUTS = VALID_OUTPUTS + VALID;

    // Add selected box to scores
    scores.push_back(edsl::cast(CURRENT_NODE, thres_type_));
    SCORE = edsl::select(SCORE > 0.0f, SCORE, NEG1);
    scores.push_back(edsl::cast(SCORE, thres_type_));

    // Set scores of current box and boxes which have IOU larger than threshold to zero
    // The scores can be same, use scatter to update current node
    edsl::Tensor NEW_SCORES_UPDATE =
        edsl::scatter(NEW_SCORES, CANDIDATE_INDEX, ZERO_SCATTER).axis(2).mode(edsl::ScatterMode::UPDATE_ELT);
    NEW_SCORES = edsl::reshape(NEW_SCORES_UPDATE, {num_batches, num_classes, num_boxes});

    edsl::Tensor IOU_CANDIDATE = edsl::gather(IOU, CANDIDATE_INDEX).mode(edsl::GatherMode::ND).batchDims(1);
    // use >= to include suppose_hard_suppresion case
    NEW_SCORES = edsl::select(IOU_CANDIDATE >= IOU_THRESHOLD_I, ZERO, NEW_SCORES);

    // Add selected box to boxes
    edsl::Tensor BOX_INDEX = edsl::select(SCORE > 0.0f, edsl::cast(CANDIDATE_INDEX, box_output_type_), NEG1_O);
    boxes.push_back(CURRENT_NODE);
    boxes.push_back(edsl::reshape(BOX_INDEX, {num_batches, num_classes, 1}));

    // update scores for current class
    edsl::Tensor SCALE = edsl::exp(IOU_CANDIDATE * IOU_CANDIDATE * WEIGHT);
    NEW_SCORES = NEW_SCORES * SCALE;
    NEW_SCORES = edsl::select(NEW_SCORES > SCORE_THRESHOLD_I, NEW_SCORES, ZERO);
  }

  VALID_OUTPUTS = edsl::cast(VALID_OUTPUTS, box_output_type_);

  int num_results = num_batches * num_classes * num_boxes_per_class;
  // concatenate scores
  edsl::Tensor SCORES_RESULT = edsl::reshape(op::concatenate(scores, 2), {num_results, 3});
  // concatenate boxes
  edsl::Tensor BOXES_RESULT = edsl::reshape(op::concatenate(boxes, 2), {num_results, 3});

  if (sort_result_descending_) {
    // Sort across batch
    edsl::Tensor SCORES_SLICE = edsl::gather(SCORES_RESULT, ONE_INT + 1).axis(1);
    SCORES_SLICE = edsl::reshape(SCORES_SLICE, {edsl::TensorDim(num_results)});
    edsl::Tensor INDEXES = edsl::argsort(SCORES_SLICE, 0, edsl::SortDirection::DESC);

    SCORES_RESULT = edsl::gather(SCORES_RESULT, INDEXES).axis(0);
    BOXES_RESULT = edsl::gather(BOXES_RESULT, INDEXES).axis(0);
  } else {
    // Put all -1 to the end, we now just have -1 at end of each class
    edsl::TensorDim dst(num_results * 3);
    auto INDEX = edsl::index({dst}, 0);
    auto MAX = edsl::cast(edsl::Tensor{dst}, DType::INT32);
    SCORES_RESULT = edsl::reshape(SCORES_RESULT, {dst});
    BOXES_RESULT = edsl::reshape(BOXES_RESULT, {dst});
    edsl::Tensor NEG1_THRES = edsl::cast(edsl::Tensor(-1), thres_type_);
    auto INDEX2 = edsl::select(SCORES_RESULT != NEG1_THRES, INDEX, MAX);
    auto INDEX3 = edsl::argsort(edsl::cast(INDEX2, DType::FLOAT32), 0);
    SCORES_RESULT = edsl::gather(SCORES_RESULT, INDEX3).axis(0);
    BOXES_RESULT = edsl::gather(BOXES_RESULT, INDEX3).axis(0);
    SCORES_RESULT = edsl::reshape(SCORES_RESULT, {num_results, 3});
    BOXES_RESULT = edsl::reshape(BOXES_RESULT, {num_results, 3});
  }

  return {BOXES_RESULT, SCORES_RESULT, VALID_OUTPUTS};
}

void registerNonMaxSuppression() {
  registerOp("NonMaxSuppression", [](const Context& ctx) {
    auto* layer = ngraph::as_type<NonMaxSuppression>(ctx.layer);
    // OPSET5 provides multiple templates, follow setup() choice now.
    IE_ASSERT(ctx.operands.size() == 6);
    auto BOXES = ctx.operands.at(0);
    auto SCORES = ctx.operands.at(1);
    auto MAX_OUTPUT_BOXES_PER_CLASS = ctx.operands.at(2);
    auto IOU_THRESHOLD = ctx.operands.at(3);
    auto SCORE_THRESHOLD = ctx.operands.at(4);
    auto SOFT_NMS_SIGMA = ctx.operands.at(5);
    bool center_point_box =
        layer->get_box_encoding() == ngraph::op::v5::NonMaxSuppression::BoxEncodingType::CENTER ? true : false;
    bool sort_result_descending = layer->get_sort_result_descending();

    auto* max_output_boxes_per_class_op = ngraph::as_type<ngraph::op::Constant>(ctx.layer->get_input_node_ptr(2));
    if (max_output_boxes_per_class_op == nullptr) {
      THROW_IE_EXCEPTION << "Dynamic output size for non_max_suppression not supported by PlaidML plugin now";
    }
    int32_t max_output_boxes_per_class = max_output_boxes_per_class_op->get_vector<int>()[0];

    DType box_input_type = to_plaidml(layer->get_input_element_type(0));
    DType box_output_type = to_plaidml(layer->get_output_type());
    DType thres_type = to_plaidml(layer->get_input_element_type(3));

    std::vector<edsl::Tensor> OUTPUTS =
        nms(BOXES, SCORES, max_output_boxes_per_class, IOU_THRESHOLD, SCORE_THRESHOLD, SOFT_NMS_SIGMA)
            .center_point_box(center_point_box)
            .sort_result_descending(sort_result_descending)
            .box_input_type(box_input_type)
            .box_output_type(box_output_type)
            .thres_type(thres_type)
            .build();

    return edsl::make_tuple(OUTPUTS);
  });
}

}  // namespace PlaidMLPlugin
