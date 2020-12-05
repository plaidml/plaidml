// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// NB: IE_SET_METRIC requires this header to be included first
#include "ie_metric_helpers.hpp"

#include "plaidml_executable_network.hpp"

#include <fstream>  // NOLINT[build/include_order]
#include <memory>   // NOLINT[build/include_order]
#include <vector>   // NOLINT[build/include_order]

#include "plaidml_builder.hpp"
#include "plaidml_infer_request.hpp"

using namespace InferenceEngine;

namespace PlaidMLPlugin {

static plaidml::Program buildProgram(const ICNNNetwork& network) {
  InputsDataMap inputsInfo;
  network.getInputsInfo(inputsInfo);

  OutputsDataMap outputsInfo;
  network.getOutputsInfo(outputsInfo);

  std::shared_ptr<const ngraph::Function> func = network.getFunction();
  return buildProgram(func, network.getName(), inputsInfo, outputsInfo);
}

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

void PlaidMLExecutableNetwork::ExportImpl(std::ostream& model) {  //
  model << program_.str() << std::endl;
}

void PlaidMLExecutableNetwork::Export(const std::string& modelFileName) {
  std::ofstream modelFile(modelFileName, std::ios::out);
  if (modelFile.is_open()) {
    ExportImpl(modelFile);
  } else {
    THROW_IE_EXCEPTION << "The " << modelFileName << " file can not be opened for export";
  }
}

}  // namespace PlaidMLPlugin
