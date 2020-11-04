// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "ie_icnn_network.hpp"

#include "ngraph/node.hpp"

#include "plaidml/edsl/edsl.h"

namespace PlaidMLPlugin {

class PlaidMLProgramBuilder {
 public:
  PlaidMLProgramBuilder() = delete;
  explicit PlaidMLProgramBuilder(const InferenceEngine::ICNNNetwork& network);

  plaidml::Program program();

 private:
  void handleConstant(const std::shared_ptr<ngraph::Node>& node);
  void handleParameter(const std::shared_ptr<ngraph::Node>& node);
  void handleOutput(const std::shared_ptr<ngraph::Node>& node);
  void handleOp(const std::shared_ptr<ngraph::Node>& node);

  std::shared_ptr<const ngraph::Function> fcn_;
  InferenceEngine::InputsDataMap networkInputs_;
  InferenceEngine::OutputsDataMap networkOutputs_;
  bool graph_parsed = false;

  // Lets us look up the PlaidML tensor by the name of the node that produces it and the index of which output it is
  std::map<std::pair<std::string, size_t>, plaidml::edsl::Tensor> tensorMap_;

  // Go from the names OV uses for a networks inputs and outputs to the corresponding PlaidML Tensor
  std::map<std::string, plaidml::edsl::Tensor> tensorIONameMap_;
};

plaidml::Program makePlaidMLProgram(const InferenceEngine::ICNNNetwork& network);

}  // namespace PlaidMLPlugin
