// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// NB: IE_SET_METRIC requires this header to be included first
#include "ie_metric_helpers.hpp"

#include "plaidml_executable_network.hpp"

#include <memory>  // NOLINT[build/include_order]
#include <vector>  // NOLINT[build/include_order]

#include "plaidml_builder.hpp"
#include "plaidml_infer_request.hpp"

using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

InferRequestInternal::Ptr PlaidMLExecutableNetwork::CreateInferRequestImpl(InputsDataMap networkInputs,
                                                                           OutputsDataMap networkOutputs) {
  return std::make_shared<PlaidMLInferRequest>(networkInputs, networkOutputs, program_);
}

PlaidMLExecutableNetwork::PlaidMLExecutableNetwork(const ICNNNetwork& network, const std::string& device)
    : program_(buildProgram(network)) {
  program_.compile();
}

void PlaidMLExecutableNetwork::GetMetric(const std::string& name, Parameter& result, ResponseDesc* resp) const {
  if (name == METRIC_KEY(SUPPORTED_METRICS)) {
    std::vector<std::string> metrics = {
        METRIC_KEY(SUPPORTED_METRICS),
        METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS),
    };
    result = IE_SET_METRIC(SUPPORTED_METRICS, metrics);
  } else if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
    result = IE_SET_METRIC(OPTIMAL_NUMBER_OF_INFER_REQUESTS, 1);
  } else {
    THROW_IE_EXCEPTION << "Unsupported ExecutableNetwork metric: " << name;
  }
}

}  // namespace PlaidMLPlugin
