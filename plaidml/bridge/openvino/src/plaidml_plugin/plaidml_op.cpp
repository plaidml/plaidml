// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <string>
#include <vector>

#include <ie_layers.h>

#include "plaidml_op.hpp"
#include "plaidml_util.hpp"

#include "plaidml_op_clamp.hpp"
#include "plaidml_op_concat.hpp"
#include "plaidml_op_convolution.hpp"
#include "plaidml_op_crop.hpp"
#include "plaidml_op_eltwise.hpp"
#include "plaidml_op_fc.hpp"
#include "plaidml_op_norm.hpp"
#include "plaidml_op_permute.hpp"
#include "plaidml_op_pooling.hpp"
#include "plaidml_op_power.hpp"
#include "plaidml_op_relu.hpp"
#include "plaidml_op_reshape.hpp"
#include "plaidml_op_scaleshift.hpp"
#include "plaidml_op_sigmoid.hpp"
#include "plaidml_op_softmax.hpp"

using namespace PlaidMLPlugin;

std::shared_ptr<Op> Op::Create(const InferenceEngine::CNNLayerPtr& layer) {
  auto type = util::str_to_type(layer->type);
  std::shared_ptr<Op> op;

  switch (type) {
    case util::OpType::Convolution:
      op = std::make_shared<OpConvolution>();
      break;
    case util::OpType::Pooling:
      op = std::make_shared<OpPooling>();
      break;
    case util::OpType::ScaleShift:
      op = std::make_shared<OpScaleShift>();
      break;
    case util::OpType::ReLU:
      op = std::make_shared<OpReLU>();
      break;
    // case util::OpType::Eltwise:
    //  op = std::make_shared<OpEltwise>();
    //  break;
    case util::OpType::Reshape:
      op = std::make_shared<OpReshape>();
      break;
    // case util::OpType::Fc:
    // op = std::make_shared<OpFc>();
    //  break;
    case util::OpType::Softmax:
      op = std::make_shared<OpSoftmax>();
      break;
    case util::OpType::Power:
      op = std::make_shared<OpPower>();
      break;
    case util::OpType::Clamp:
      op = std::make_shared<OpClamp>();
      break;
    case util::OpType::Permute:
      op = std::make_shared<OpPermute>();
      break;
    case util::OpType::Concat:
      op = std::make_shared<OpConcat>();
      break;
    // case util::OpType::Sigmoid:
    //  op = std::make_shared<OpSigmoid>();
    //  break;
    case util::OpType::Crop:
      op = std::make_shared<OpCrop>();
      break;
    case util::OpType::Norm:
      op = std::make_shared<OpNorm>();
      break;
    default:
      THROW_IE_EXCEPTION << "Layer " << layer->type << " not supported ";
  }

  op->Init(layer);
  return op;
}

void Op::Apply(State* state) {
  LoadWeights(state);

  PackInputs(state);
  PackWeights(state);
  PackOutputs(state);

  Execute();
}

void Op::Init(const CNNLayerPtr& layer) { layer_ = layer; }

void Op::LoadWeights(State* state) { /* do nothing by default */
}

void Op::PackInputs(State* state) {
  for (const auto in_data : layer_->insData) {
    const auto& input_name = in_data.lock()->getName();
    ctx_.inputs_.emplace_back(state->slot<plaidml::edsl::Tensor>()[input_name]);
  }
}

void Op::PackWeights(State* state) {
  const auto& bindings = state->slot<std::vector<plaidml::exec::Binding>>();
  auto it = bindings.find(layer_->name);
  if (it != bindings.end()) {
    for (const auto& binding : it->second) {
      ctx_.inputs_.emplace_back(binding.tensor);
    }
  }
}

void Op::PackOutputs(State* state) {
  // FIXME Now works only with single output
  ctx_.outputs_.emplace_back(&state->slot<plaidml::edsl::Tensor>()[layer_->name]);
}

void Op::Execute() { THROW_IE_EXCEPTION << "Default implementation of Execute is called for layer: " << layer_->name; }
