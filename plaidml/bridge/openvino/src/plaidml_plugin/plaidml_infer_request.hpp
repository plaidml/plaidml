// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_map>

#include "plaidml_executable_network.hpp"
#include "plaidml_state.hpp"

namespace PlaidMLPlugin {

class PlaidMLInferRequest : public InferenceEngine::InferRequestInternal {
 public:
  using Ptr = std::shared_ptr<PlaidMLInferRequest>;

  explicit PlaidMLInferRequest(InferenceEngine::InputsDataMap networkInputs,
                               InferenceEngine::OutputsDataMap networkOutputs, const std::shared_ptr<State>& state);

  void InferImpl() override;

  void GetPerformanceCounts(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& perfMap) const override;

 private:
  void AllocateInputs();
  void AllocateOutputs();
  void SyncInput();
  void SyncOutput();

 private:
  std::unique_ptr<plaidml::exec::Binder> binder_;
  std::shared_ptr<plaidml::exec::Executable> exec_;
  std::shared_ptr<State> state_;
};

}  // namespace PlaidMLPlugin
