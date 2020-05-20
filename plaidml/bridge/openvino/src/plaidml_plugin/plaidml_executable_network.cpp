// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "plaidml/op/op.h"

#include <details/caseless.hpp>
#include <details/ie_cnn_network_tools.h>
#include <ie_util_internal.hpp>

#include <low_precision_transformations/concat_multi_channels.hpp>
#include <low_precision_transformations/eltwise_cpu.hpp>
#include <low_precision_transformations/fully_connected.hpp>
#include <low_precision_transformations/pooling.hpp>
#include <low_precision_transformations/transformer.hpp>

#include "plaidml_executable_network.hpp"
#include "plaidml_infer_request.hpp"
#include "plaidml_op.hpp"
#include "plaidml_util.hpp"

using namespace InferenceEngine;
using namespace PlaidMLPlugin;

InferenceEngine::InferRequestInternal::Ptr PlaidMLExecutableNetwork::CreateInferRequestImpl(
    InferenceEngine::InputsDataMap networkInputs, InferenceEngine::OutputsDataMap networkOutputs) {
  return std::make_shared<PlaidMLInferRequest>(networkInputs, networkOutputs, state_);
}

PlaidMLExecutableNetwork::PlaidMLExecutableNetwork(InferenceEngine::ICNNNetwork& network,
                                                   const std::string& configuration_type)
    : state_(new State(configuration_type)) {
  //  Apply low precision graph transformations
  // using namespace details;
  auto params = InferenceEngine::details::LayerTransformation::Params(
      true,  // updatePrecisions
      true,  // quantizeOutputs
      true,  // weightsToConst
      InferenceEngine::details::LayerTransformation::QuantizedTensorAlignment::
          UpdateLevel,  // quantizedTensorAlignmentOnActivations
      InferenceEngine::details::LayerTransformation::QuantizedTensorAlignment::
          None,  // quantizedTensorAlignmentOnWeights
      true,      // roundQuantizedValues
      false,     // updateBiases
      false);    // supportAsymmetricQuantization

  /* TODO: Low Precision Graph Transformations do not set precision on conv edges, activate this code when fixed or
  implement W/A details::LowPrecisionTransformer transformer(LowPrecisionTransformer::getAllTransformations(params));
  transformer.transform(network);
  ResponseDesc resp;
  network.serialize("./lowp_transformed.xml", "./lowp_transformed.bin", &resp); */

  InitInputs(network);
  plaidml::op::init();

  auto sorted_layers = CNNNetSortTopologically(network);
  for (auto& layer : sorted_layers) {
    if (InferenceEngine::details::CaselessEq<std::string>()(layer->type, "input")) continue;
    Op::Create(layer)->Apply(state_.get());
  }
}

void PlaidMLExecutableNetwork::InitInputs(InferenceEngine::ICNNNetwork& network) {
  InputsDataMap input_info;
  network.getInputsInfo(input_info);

  // NB: Create placeholders for inputs
  for (const auto& info : input_info) {
    auto desc = info.second->getTensorDesc();
    if (desc.getLayout() == Layout::NCHW) {
      const auto& dims = desc.getDims();
      desc.reshape({dims[0], dims[2], dims[3], dims[1]});
    }
    auto ph = plaidml::edsl::Placeholder(util::to_plaidml(desc));
    state_->slot<plaidml::edsl::Tensor>().emplace(info.first, std::move(ph));
  }
}
