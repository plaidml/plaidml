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
  explicit nms(edsl::Tensor Boxes, edsl::Tensor Scores, edsl::Tensor IOU_threshold, edsl::Tensor Score_threshold,
               int max_output_boxes_per_class)
      : Boxes_(Boxes),
        Scores_(Scores),
        max_output_boxes_per_class_(max_output_boxes_per_class),
        IOU_threshold_(IOU_threshold),
        Score_threshold_(Score_threshold),
        soft_nms_sigma_(0.0f),
        center_point_box_(false),
        sort_result_descending_(false),
        box_input_type_(DType::FLOAT32),
        box_output_type_(DType::INT32),
        thres_type_(DType::FLOAT32) {}

  nms& soft_nms_sigma(float soft_nms_sigma) {
    soft_nms_sigma_ = soft_nms_sigma;
    return *this;
  }

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
  edsl::Tensor getIOU();

  edsl::Tensor Boxes_;
  edsl::Tensor Scores_;
  int max_output_boxes_per_class_;
  edsl::Tensor IOU_threshold_;
  edsl::Tensor Score_threshold_;
  float soft_nms_sigma_;
  bool center_point_box_;
  bool sort_result_descending_;
  DType box_input_type_;
  DType box_output_type_;
  DType thres_type_;
};

// Get intersection over union of boxes
// The shape of the result tensor is [num_batches, num_boxes, num_boxes]
// IOU[batch, box_1, box_2] is the IOU of box_1 and box_2 for the image batch.
edsl::Tensor nms::getIOU() {
  edsl::Tensor Boxes_y1;
  edsl::Tensor Boxes_x1;
  edsl::Tensor Boxes_y2;
  edsl::Tensor Boxes_x2;
  // The memref is used for gather
  edsl::Tensor Zero = edsl::cast(edsl::index({edsl::TensorDim(1)}, 0), box_input_type_);
  edsl::Tensor Zero_int = edsl::cast(Zero, DType::INT32);
  if (center_point_box_) {
    // The box data is [x, y, width, height] in center mode.
    edsl::Tensor Boxes_xcenter = edsl::gather(Boxes_, Zero_int).axis(2);
    edsl::Tensor Boxes_ycenter = edsl::gather(Boxes_, Zero_int + 1).axis(2);
    edsl::Tensor Boxes_width_half = edsl::gather(Boxes_, Zero_int + 2).axis(2);
    Boxes_width_half = Boxes_width_half / 2.0f;
    edsl::Tensor Boxes_height_half = edsl::gather(Boxes_, Zero_int + 3).axis(2);
    Boxes_height_half = Boxes_height_half / 2.0f;
    Boxes_x1 = Boxes_xcenter - Boxes_width_half;
    Boxes_x2 = Boxes_xcenter + Boxes_width_half;
    Boxes_y1 = Boxes_ycenter - Boxes_height_half;
    Boxes_y2 = Boxes_ycenter + Boxes_height_half;
  } else {
    // The box data is [y1, x1, y2, x2] in corner mode
    Boxes_y1 = edsl::gather(Boxes_, Zero_int).axis(2);
    Boxes_x1 = edsl::gather(Boxes_, Zero_int + 1).axis(2);
    Boxes_y2 = edsl::gather(Boxes_, Zero_int + 2).axis(2);
    Boxes_x2 = edsl::gather(Boxes_, Zero_int + 3).axis(2);
  }

  // Boxes dims is [num_batches, num_boxes, 4]
  std::vector<edsl::TensorDim> dims(Boxes_.rank());
  Boxes_.bind_dims(dims);
  edsl::TensorDim num_batches = dims[0];
  edsl::TensorDim num_boxes = dims[1];
  edsl::TensorDim one(1);

  // Create IOU of each box pair.
  Boxes_y1 = edsl::reshape(Boxes_y1, {num_batches, num_boxes});
  Boxes_x1 = edsl::reshape(Boxes_x1, {num_batches, num_boxes});
  Boxes_y2 = edsl::reshape(Boxes_y2, {num_batches, num_boxes});
  Boxes_x2 = edsl::reshape(Boxes_x2, {num_batches, num_boxes});

  // Calc box local area
  edsl::Tensor IOU_areai = (Boxes_y2 - Boxes_y1) * (Boxes_x2 - Boxes_x1);
  // Calc intersecion area of box pair
  edsl::Tensor IOU_intersection_ymin = op::maximum(edsl::reshape(Boxes_y1, {num_batches, num_boxes, one}),
                                                   edsl::reshape(Boxes_y1, {num_batches, one, num_boxes}));
  edsl::Tensor IOU_intersection_xmin = op::maximum(edsl::reshape(Boxes_x1, {num_batches, num_boxes, one}),
                                                   edsl::reshape(Boxes_x1, {num_batches, one, num_boxes}));
  edsl::Tensor IOU_intersection_ymax = op::minimum(edsl::reshape(Boxes_y2, {num_batches, num_boxes, one}),
                                                   edsl::reshape(Boxes_y2, {num_batches, one, num_boxes}));
  edsl::Tensor IOU_intersection_xmax = op::minimum(edsl::reshape(Boxes_x2, {num_batches, num_boxes, one}),
                                                   edsl::reshape(Boxes_x2, {num_batches, one, num_boxes}));
  edsl::Tensor IOU_intersection_area_ygap = IOU_intersection_ymax - IOU_intersection_ymin;
  edsl::Tensor IOU_intersection_area_xgap = IOU_intersection_xmax - IOU_intersection_xmin;
  edsl::Tensor IOU_intersection_area =
      edsl::select(IOU_intersection_area_ygap > 0.0f, IOU_intersection_area_ygap, Zero) *
      edsl::select(IOU_intersection_area_xgap > 0.0f, IOU_intersection_area_xgap, Zero);
  // Calc IOU of box pair
  edsl::Tensor IOU_denominator =
      op::unsqueeze(IOU_areai, {-1}) + op::unsqueeze(IOU_areai, {-2}) - IOU_intersection_area;
  edsl::Tensor IOU_denominator_zeroed = edsl::select(IOU_denominator <= 0.0f, Zero, 1.0f / IOU_denominator);
  edsl::Tensor IOU = IOU_intersection_area * IOU_denominator_zeroed;
  return IOU;
}

std::vector<edsl::Tensor> nms::build() {
  std::vector<int64_t> scores_shape = Scores_.compute_shape().sizes();
  int boxes_count = scores_shape[2];

  std::vector<edsl::TensorDim> Scores_dims(Scores_.rank());
  Scores_.bind_dims(Scores_dims);
  edsl::TensorDim num_batches = Scores_dims[0];
  edsl::TensorDim num_classes = Scores_dims[1];
  edsl::TensorDim num_boxes = Scores_dims[2];
  edsl::TensorDim one(1);

  // Used for gather and select
  edsl::Tensor Zero = edsl::cast(edsl::index({one}, 0), box_input_type_);
  edsl::Tensor Zero_int = edsl::cast(Zero, DType::INT32);
  edsl::Tensor One = Zero + 1;
  edsl::Tensor One_int = Zero_int + 1;
  edsl::Tensor NEG1 = -One;
  edsl::Tensor Neg1_o = edsl::cast(NEG1, box_output_type_);

  std::vector<edsl::Tensor> boxes;
  std::vector<edsl::Tensor> scores;
  edsl::Tensor Valid_outputs = Zero;
  edsl::Tensor IOU = getIOU();

  float weight = 0.0f;
  if (soft_nms_sigma_ != 0) {
    weight = -0.5 / soft_nms_sigma_;
  }

  TensorShape Node_shape(DType::FLOAT32, {1, 1, 2});
  std::vector<float> invalid_node_index = {-1, -1};
  Buffer buffer_invalid_node(Node_shape);
  buffer_invalid_node.copy_from(invalid_node_index.data());
  auto Invalid_node = edsl::cast(edsl::Constant(buffer_invalid_node, "Invalid_node"), box_output_type_);

  int num_boxes_per_class = boxes_count;
  if (max_output_boxes_per_class_ != 0) {
    num_boxes_per_class = std::min(boxes_count, max_output_boxes_per_class_);
  }

  auto Index_b = edsl::index({num_batches, num_classes, one}, 0);
  auto Index_c = edsl::index({num_batches, num_classes, one}, 1);
  auto Index_bc = edsl::cast(op::concatenate({Index_b, Index_c}, 2), box_output_type_);

  edsl::Tensor Score_threshold_i = edsl::cast(Score_threshold_, box_input_type_);
  edsl::Tensor IOU_threshold_i = edsl::cast(IOU_threshold_, box_input_type_);
  edsl::Tensor New_scores = edsl::select(Scores_ > Score_threshold_i, Scores_, Zero);

  // std::vector<edsl::TensorDim> Scatter_dims = {num_batches, num_classes, one};
  // std::vector<edsl::TensorIndex> Scatter_idxs(3);
  // edsl::Tensor Zero_scatter = edsl::Contraction(Scatter_dims, Scatter_idxs).assign(Zero(0));
  edsl::Tensor Zero_scatter = op::broadcast(Zero, {2, 3, 1}, {0});
  // Select box
  for (int k = 0; k < num_boxes_per_class; k++) {
    // Select the box with largest score
    edsl::Tensor Candidate_index =
        edsl::gather(edsl::argsort(New_scores, 2, edsl::SortDirection::DESC), Zero_int).axis(2);
    edsl::Tensor SCORE = edsl::reshape(op::max(New_scores, edsl::Value(2)), {num_batches, num_classes, one});
    edsl::Tensor Current_node = edsl::select(SCORE > 0.0f, Index_bc, Invalid_node);

    // Update count of selected box
    edsl::Tensor Valid = op::sum(edsl::select(SCORE != 0.0f, One, Zero));
    Valid_outputs = Valid_outputs + Valid;

    // Add selected box to scores
    scores.push_back(edsl::cast(Current_node, thres_type_));
    SCORE = edsl::select(SCORE > 0.0f, SCORE, NEG1);
    scores.push_back(edsl::cast(SCORE, thres_type_));

    // Set scores of current box and boxes which have IOU larger than threshold to zero
    // The scores can be same, use scatter to update current node
    edsl::Tensor New_scores_update =
        edsl::scatter(New_scores, Candidate_index, Zero_scatter).axis(2).mode(edsl::ScatterMode::UPDATE_ELT);
    New_scores = edsl::reshape(New_scores_update, {num_batches, num_classes, num_boxes});

    edsl::Tensor IOU_candidate = edsl::gather(IOU, Candidate_index).mode(edsl::GatherMode::ND).batchDims(1);
    // use >= to include suppose_hard_suppresion case
    New_scores = edsl::select(IOU_candidate >= IOU_threshold_i, Zero, New_scores);

    // Add selected box to boxes
    edsl::Tensor Box_index = edsl::select(SCORE > 0.0f, edsl::cast(Candidate_index, box_output_type_), Neg1_o);
    boxes.push_back(Current_node);
    boxes.push_back(edsl::reshape(Box_index, {num_batches, num_classes, one}));

    // update scores for current class
    edsl::Tensor Scale = edsl::exp(IOU_candidate * IOU_candidate * weight);
    New_scores = New_scores * Scale;
    New_scores = edsl::select(New_scores > Score_threshold_i, New_scores, Zero);
  }

  Valid_outputs = edsl::cast(Valid_outputs, box_output_type_);

  edsl::TensorDim num_results = num_batches * num_classes * num_boxes_per_class;
  // concatenate scores
  edsl::Tensor Scores_result = edsl::reshape(op::concatenate(scores, 2), {num_results, edsl::TensorDim(3)});
  // concatenate boxes
  edsl::Tensor Boxes_result = edsl::reshape(op::concatenate(boxes, 2), {num_results, edsl::TensorDim(3)});

  if (sort_result_descending_) {
    // Sort across batch
    edsl::Tensor Scores_slice = edsl::gather(Scores_result, One_int + 1).axis(1);
    Scores_slice = edsl::reshape(Scores_slice, {num_results});
    edsl::Tensor Indexes = edsl::argsort(Scores_slice, 0, edsl::SortDirection::DESC);

    Scores_result = edsl::gather(Scores_result, Indexes).axis(0);
    Boxes_result = edsl::gather(Boxes_result, Indexes).axis(0);
  } else {
    // Put all -1 to the end, we now just have -1 at end of each class
    edsl::TensorDim dst = num_results * 3;
    auto Index = edsl::index({dst}, 0);
    auto Max = edsl::cast(edsl::Tensor{dst}, DType::INT32);
    Scores_result = edsl::reshape(Scores_result, {dst});
    Boxes_result = edsl::reshape(Boxes_result, {dst});
    edsl::Tensor Neg1_thres = edsl::cast(edsl::Tensor(-1), thres_type_);
    auto Index2 = edsl::select(Scores_result != Neg1_thres, Index, Max);
    auto Index3 = edsl::argsort(edsl::cast(Index2, DType::FLOAT32), 0);
    Scores_result = edsl::gather(Scores_result, Index3).axis(0);
    Boxes_result = edsl::gather(Boxes_result, Index3).axis(0);
    Scores_result = edsl::reshape(Scores_result, {num_results, edsl::TensorDim(3)});
    Boxes_result = edsl::reshape(Boxes_result, {num_results, edsl::TensorDim(3)});
  }

  return {Boxes_result, Scores_result, Valid_outputs};
}

void registerNonMaxSuppression() {
  registerOp("NonMaxSuppression", [](const Context& ctx) {
    auto* layer = ngraph::as_type<NonMaxSuppression>(ctx.layer);
    // OPSET5 provides multiple templates, follow setup() choice now.
    IE_ASSERT(ctx.operands.size() == 6);
    auto Boxes = ctx.operands.at(0);
    auto Scores = ctx.operands.at(1);
    int32_t max_output_boxes_per_class = layer->max_boxes_output_from_input();
    auto IOU_threshold = ctx.operands.at(3);
    auto Score_threshold = ctx.operands.at(4);
    float soft_nms_sigma = layer->soft_nms_sigma_from_input();
    bool center_point_box =
        layer->get_box_encoding() == ngraph::op::v5::NonMaxSuppression::BoxEncodingType::CENTER ? true : false;
    bool sort_result_descending = layer->get_sort_result_descending();

    DType box_input_type = to_plaidml(layer->get_input_element_type(0));
    DType box_output_type = to_plaidml(layer->get_output_type());
    DType thres_type = to_plaidml(layer->get_input_element_type(3));

    std::vector<edsl::Tensor> Outputs = nms(Boxes, Scores, IOU_threshold, Score_threshold, max_output_boxes_per_class)
                                            .soft_nms_sigma(soft_nms_sigma)
                                            .center_point_box(center_point_box)
                                            .sort_result_descending(sort_result_descending)
                                            .box_input_type(box_input_type)
                                            .box_output_type(box_output_type)
                                            .thres_type(thres_type)
                                            .build();

    return edsl::make_tuple(Outputs);
  });
}

}  // namespace PlaidMLPlugin
