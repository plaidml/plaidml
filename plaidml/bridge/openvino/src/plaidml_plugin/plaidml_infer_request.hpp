// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>

#include "cpp_interfaces/impl/ie_infer_request_internal.hpp"

#include "ngraph/descriptor/tensor.hpp"

#include "plaidml/edsl/edsl.h"
#include "plaidml/exec/exec.h"

namespace PlaidMLPlugin {

class PlaidMLInferRequest : public InferenceEngine::InferRequestInternal {
 public:
  using Ptr = std::shared_ptr<PlaidMLInferRequest>;

  explicit PlaidMLInferRequest(const InferenceEngine::InputsDataMap& networkInputs,
                               const InferenceEngine::OutputsDataMap& networkOutputs,
                               const plaidml::edsl::Program& program,
                               const std::map<std::string, plaidml::edsl::Tensor>& tensorIONameMap);

  void InferImpl() override;

  void GetPerformanceCounts(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& perfMap) const override;

 private:
  void AllocateInputs();
  void AllocateOutputs();
  void SyncInput();
  void SyncOutput();

 private:
  std::map<std::string, plaidml::edsl::Tensor> tensorIONameMap_;
  plaidml::exec::Binder binder_;
  std::shared_ptr<plaidml::exec::Executable> exec_;
};

}  // namespace PlaidMLPlugin
