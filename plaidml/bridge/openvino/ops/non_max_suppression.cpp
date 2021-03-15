// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

#include "ngraph/opsets/opset5.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;  // NOLINT[build/namespaces]
using edsl::Tensor;
using ngraph::opset5::NonMaxSuppression;

namespace PlaidMLPlugin {

std::vector<Tensor> NMS(Tensor BOXES, Tensor SCORES, int32_t max_output_boxes_per_class, Tensor IOU_THRESHOLD,
                        Tensor SCORE_THRESHOLD, Tensor SOFT_NMS_SIGMA, bool center_point_box,
                        bool sort_result_descending, DType box_input_type, DType box_output_type, DType thres_type) {
  std::vector<int64_t> boxes_shape = BOXES.compute_shape().sizes();
  std::vector<int64_t> scores_shape = SCORES.compute_shape().sizes();
  int num_batches = boxes_shape[0];
  int num_boxes = boxes_shape[1];
  int box_size = 4;
  int num_classes = scores_shape[1];
  Tensor ZERO = edsl::cast(Tensor(0), box_input_type);
  Tensor ZERO_THRES = edsl::cast(Tensor(0), thres_type);
  Tensor ONE_THRES = edsl::cast(Tensor(1), thres_type);
  Tensor NEG1 = edsl::cast(Tensor(-1), box_input_type);
  Tensor NEG1_OUTPUT = edsl::cast(Tensor(-1), box_output_type);

  std::vector<Tensor> boxes;
  std::vector<Tensor> scores;
  Tensor VALID_OUTPUTS = ZERO;

  Tensor BOXES_Y1;
  Tensor BOXES_X1;
  Tensor BOXES_Y2;
  Tensor BOXES_X2;
  if (center_point_box) {
    Tensor BOXES_XCENTER = op::slice(BOXES).add_dim(0, num_batches).add_dim(0, num_boxes).add_dim(0, 1);
    Tensor BOXES_YCENTER = op::slice(BOXES).add_dim(0, num_batches).add_dim(0, num_boxes).add_dim(1, 2);
    Tensor BOXES_WIDTH_HALF = op::slice(BOXES).add_dim(0, num_batches).add_dim(0, num_boxes).add_dim(2, 3);
    BOXES_WIDTH_HALF = BOXES_WIDTH_HALF / 2.0f;
    Tensor BOXES_HEIGHT_HALF = op::slice(BOXES).add_dim(0, num_batches).add_dim(0, num_boxes).add_dim(3, box_size);
    BOXES_HEIGHT_HALF = BOXES_HEIGHT_HALF / 2.0f;
    BOXES_X1 = BOXES_XCENTER - BOXES_WIDTH_HALF;
    BOXES_X2 = BOXES_XCENTER + BOXES_WIDTH_HALF;
    BOXES_Y1 = BOXES_YCENTER - BOXES_HEIGHT_HALF;
    BOXES_Y2 = BOXES_YCENTER + BOXES_HEIGHT_HALF;
  } else {
    BOXES_Y1 = op::slice(BOXES).add_dim(0, num_batches).add_dim(0, num_boxes).add_dim(0, 1);
    BOXES_X1 = op::slice(BOXES).add_dim(0, num_batches).add_dim(0, num_boxes).add_dim(1, 2);
    BOXES_Y2 = op::slice(BOXES).add_dim(0, num_batches).add_dim(0, num_boxes).add_dim(2, 3);
    BOXES_X2 = op::slice(BOXES).add_dim(0, num_batches).add_dim(0, num_boxes).add_dim(3, box_size);
  }

  // TensorDim num_batches_td(num_batches);
  // TensorDim num_boxes_td(num_boxes);
  // TensorDim one_td(1);

  BOXES_Y1 = edsl::reshape(BOXES_Y1, {num_batches, num_boxes});
  BOXES_X1 = edsl::reshape(BOXES_X1, {num_batches, num_boxes});
  BOXES_Y2 = edsl::reshape(BOXES_Y2, {num_batches, num_boxes});
  BOXES_X2 = edsl::reshape(BOXES_X2, {num_batches, num_boxes});

  Tensor IOU_AREAI = (BOXES_Y2 - BOXES_Y1) * (BOXES_X2 - BOXES_X1);  // num_batches * num_boxes, 1*5
  Tensor IOU_INTERSECTION_YMIN =
      op::maximum(edsl::reshape(BOXES_Y1, {num_batches, num_boxes, 1}),
                  edsl::reshape(BOXES_Y1, {num_batches, 1, num_boxes}));  // shall be num_batch * num_box * num_box
  Tensor IOU_INTERSECTION_XMIN = op::maximum(edsl::reshape(BOXES_X1, {num_batches, num_boxes, 1}),
                                             edsl::reshape(BOXES_X1, {num_batches, 1, num_boxes}));
  Tensor IOU_INTERSECTION_YMAX = op::minimum(edsl::reshape(BOXES_Y2, {num_batches, num_boxes, 1}),
                                             edsl::reshape(BOXES_Y2, {num_batches, 1, num_boxes}));
  Tensor IOU_INTERSECTION_XMAX = op::minimum(edsl::reshape(BOXES_X2, {num_batches, num_boxes, 1}),
                                             edsl::reshape(BOXES_X2, {num_batches, 1, num_boxes}));
  Tensor IOU_INTERSECTION_AREA_YGAP = IOU_INTERSECTION_YMAX - IOU_INTERSECTION_YMIN;
  Tensor IOU_INTERSECTION_AREA_XGAP = IOU_INTERSECTION_XMAX - IOU_INTERSECTION_XMIN;
  Tensor IOU_INTERSECTION_AREA = edsl::select(IOU_INTERSECTION_AREA_YGAP > 0.0f, IOU_INTERSECTION_AREA_YGAP, ZERO) *
                                 edsl::select(IOU_INTERSECTION_AREA_XGAP > 0.0f, IOU_INTERSECTION_AREA_XGAP, ZERO);

  Tensor IOU_DENOMINATOR = op::unsqueeze(IOU_AREAI, {-1}) + op::unsqueeze(IOU_AREAI, {-2}) - IOU_INTERSECTION_AREA;

  Tensor IOU_DENOMINATOR_ZEROED = edsl::select(IOU_DENOMINATOR <= 0.0f, ZERO, 1.0f / IOU_DENOMINATOR);
  Tensor IOU = IOU_INTERSECTION_AREA * IOU_DENOMINATOR_ZEROED;  // num_batches * num_boxes * num_boxes

  Tensor WEIGHT = edsl::select(SOFT_NMS_SIGMA != 0.0f, -0.5 / SOFT_NMS_SIGMA, ZERO_THRES);

  TensorShape NODE_SHAPE(DType::FLOAT32, {1, 1, 2});  // 1, 1, 2

  std::vector<float> invalid_node_index = {-1, -1};
  Buffer buffer_invalid_node(NODE_SHAPE);
  buffer_invalid_node.copy_from(invalid_node_index.data());
  auto INVALID_NODE = edsl::Constant(buffer_invalid_node, "INVALID_NODE");  // 1*1*2*fp32

  int num_boxes_per_class = std::min(num_boxes, max_output_boxes_per_class);

  for (int i = 0; i < num_batches; i++) {
    Tensor IOU_CURRENT_BATCH =
        op::slice(IOU).add_dim(i, i + 1).add_dim(0, num_boxes).add_dim(0, num_boxes);  // 1* num_boxes * num_boxes
    for (int j = 0; j < num_classes; j++) {
      Tensor SCORES_CLASS =
          op::slice(SCORES).add_dim(i, i + 1).add_dim(j, j + 1).add_dim(0, num_boxes);       // 1 * 1 * num_boxes
      Tensor NEW_SCORES = edsl::select(SCORES_CLASS > SCORE_THRESHOLD, SCORES_CLASS, ZERO);  // remove unused value

      std::vector<float> node_index = {static_cast<float>(i), static_cast<float>(j)};
      Buffer buffer_node(NODE_SHAPE);
      buffer_node.copy_from(node_index.data());
      std::string node_name = "NODE";
      Tensor NODE =
          edsl::cast(edsl::Constant(buffer_node, node_name + std::to_string(i) + std::to_string(j)), box_output_type);

      // Select box
      for (int k = 0; k < num_boxes_per_class; k++) {
        // Select the box with largest score
        Tensor CANDIDATE_INDEX = edsl::reshape(op::argmax(NEW_SCORES, edsl::Value(2)), {edsl::TensorDim(1)});  // 1*ui32
        Tensor SCORE = edsl::reshape(edsl::gather(NEW_SCORES, CANDIDATE_INDEX).axis(2), {1, 1, 1});            // 1*1*1
        Tensor CURRENT_NODE = edsl::select(SCORE > 0.0f, NODE, edsl::cast(INVALID_NODE, box_output_type));

        // Update count of selected box
        Tensor VALID = edsl::select(SCORE != 0.0f, ONE_THRES, ZERO_THRES);
        VALID_OUTPUTS = VALID_OUTPUTS + VALID;

        // Add selected box to scores
        scores.push_back(edsl::cast(CURRENT_NODE, thres_type));
        SCORE = edsl::select(SCORE > 0.0f, SCORE, NEG1);
        scores.push_back(edsl::cast(SCORE, thres_type));

        // Set scores of current box and boxes which have IOU larger than threshold to zero
        NEW_SCORES = edsl::select(NEW_SCORES == SCORE, ZERO, NEW_SCORES);
        Tensor IOU_CANDIDATE = edsl::gather(IOU_CURRENT_BATCH, CANDIDATE_INDEX).axis(1);  // 1*1*num_boxes
        // use >= to include suppose_hard_suppresion case
        NEW_SCORES = edsl::select(IOU_CANDIDATE >= IOU_THRESHOLD, ZERO, NEW_SCORES);  // 1*1*num_boxes

        // Add selected box to boxes
        Tensor BOX_INDEX = edsl::select(SCORE > 0.0f, edsl::cast(CANDIDATE_INDEX, box_output_type), NEG1_OUTPUT);
        boxes.push_back(CURRENT_NODE);
        boxes.push_back(edsl::reshape(BOX_INDEX, {1, 1, 1}));

        // update scores for current class
        Tensor SCALE = edsl::exp(IOU_CANDIDATE * IOU_CANDIDATE * WEIGHT);
        NEW_SCORES = NEW_SCORES * SCALE;
        NEW_SCORES = edsl::select(NEW_SCORES > SCORE_THRESHOLD, NEW_SCORES, ZERO);  // remove unused value
      }
    }
  }

  VALID_OUTPUTS = edsl::cast(VALID_OUTPUTS, box_output_type);

  int num_results = num_batches * num_classes * num_boxes_per_class;
  // concatenate scores
  Tensor SCORES_RESULT = edsl::reshape(op::concatenate(scores, 2), {num_results, 3});
  // concatenate boxes
  Tensor BOXES_RESULT = edsl::reshape(op::concatenate(boxes, 2), {num_results, 3});

  if (sort_result_descending) {
    // Sort across batch
    Tensor SCORES_SLICE = op::slice(SCORES_RESULT).add_dim(0, num_results).add_dim(2, 3);
    SCORES_SLICE = edsl::reshape(SCORES_SLICE, {edsl::TensorDim(num_results)});
    Tensor INDEXES = argsort(SCORES_SLICE, 0, edsl::SortDirection::DESC);

    SCORES_RESULT = edsl::gather(SCORES_RESULT, INDEXES).axis(0);
    BOXES_RESULT = edsl::gather(BOXES_RESULT, INDEXES).axis(0);
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
    auto SOFT_NMS_SIGMA = ctx.operands.at(4);
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

    std::vector<Tensor> OUTPUTS =
        NMS(BOXES, SCORES, max_output_boxes_per_class, IOU_THRESHOLD, SCORE_THRESHOLD, SOFT_NMS_SIGMA, center_point_box,
            sort_result_descending, box_input_type, box_output_type, thres_type);

    return edsl::make_tuple(OUTPUTS);
  });
}

}  // namespace PlaidMLPlugin
