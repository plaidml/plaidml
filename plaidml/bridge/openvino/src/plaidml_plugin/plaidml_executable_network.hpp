// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>

#include "cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp"

#include "ngraph/descriptor/tensor.hpp"

#include "plaidml/edsl/edsl.h"

namespace PlaidMLPlugin {

class PlaidMLExecutableNetwork : public InferenceEngine::ExecutableNetworkThreadSafeDefault {
 public:
  using Ptr = std::shared_ptr<PlaidMLExecutableNetwork>;

  PlaidMLExecutableNetwork(const InferenceEngine::ICNNNetwork& network, const std::string& device);
  virtual ~PlaidMLExecutableNetwork() = default;

  InferenceEngine::InferRequestInternal::Ptr CreateInferRequestImpl(
      InferenceEngine::InputsDataMap networkInputs, InferenceEngine::OutputsDataMap networkOutputs) override;

  void GetMetric(const std::string& name, InferenceEngine::Parameter& result,
                 InferenceEngine::ResponseDesc* resp) const override;

 private:
  // Go from the nGraph Tensor descriptors available from nGraph Nodes to the corresponding PlaidML Tensor
  std::map<std::shared_ptr<ngraph::descriptor::Tensor>, plaidml::edsl::Tensor> tensorMap_;

  // Go from the names OV uses for a networks inputs and outputs to the corresponding PlaidML Tensor
  std::map<std::string, plaidml::edsl::Tensor> tensorIONameMap_;
};

}  // namespace PlaidMLPlugin
