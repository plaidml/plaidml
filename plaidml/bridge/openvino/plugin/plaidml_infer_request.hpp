// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "cpp_interfaces/interface/ie_iinfer_request_internal.hpp"

#include "ngraph/descriptor/tensor.hpp"

#include "plaidml/core/core.h"
#include "plaidml/exec/exec.h"

namespace PlaidMLPlugin {

class PlaidMLInferRequest : public InferenceEngine::IInferRequestInternal {
 public:
  using Ptr = std::shared_ptr<PlaidMLInferRequest>;

  explicit PlaidMLInferRequest(const InferenceEngine::InputsDataMap& networkInputs,
                               const InferenceEngine::OutputsDataMap& networkOutputs, const plaidml::Program& program);

  void InferImpl() override;

  std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override;

 private:
  void AllocateInputs();
  void AllocateOutputs();
  void SyncInput();
  void SyncOutput();

 private:
  plaidml::Program program_;
  std::shared_ptr<plaidml::exec::Executable> exec_;
  std::vector<plaidml::Buffer> input_buffers_;
  std::vector<plaidml::Buffer> output_buffers_;
};

}  // namespace PlaidMLPlugin
