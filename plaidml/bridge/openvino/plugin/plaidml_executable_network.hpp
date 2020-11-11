// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp"

#include "plaidml/core/core.h"

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
  plaidml::Program program_;
};

}  // namespace PlaidMLPlugin
