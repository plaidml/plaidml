// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp"

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
  // Maps the names of the nGraph tensors to the corresponding PlaidML Tensors
  std::unordered_map<std::string, plaidml::edsl::Tensor> tensorMap_;
  // Maps the friendly names of the nGraph nodes generating input or output tensors to the corresponding PlaidML Tensors
  // Since general nGraph Nodes may have multiple outputs, this cannot be the same as tensorMap_; I'm also concerned
  // about possibly non-unique friendly names. However, we need to track I/O tensors by Nodes' friendly names since
  // that is was the InputsDataMap and OutputsDataMap use.
  std::unordered_map<std::string, plaidml::edsl::Tensor> tensorIOMap_;
};

}  // namespace PlaidMLPlugin
