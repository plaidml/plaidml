// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// NB: IE_SET_METRIC requires this header to be included first
#include "ie_metric_helpers.hpp"

#include "plaidml_executable_network.hpp"

#include <vector>

#include "details/ie_cnn_network_tools.h"

#include "plaidml/op/op.h"
#include "pmlc/util/logging.h"

#include "plaidml_infer_request.hpp"
#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

InferRequestInternal::Ptr PlaidMLExecutableNetwork::CreateInferRequestImpl(InputsDataMap networkInputs,
                                                                           OutputsDataMap networkOutputs) {
  IVLOG(1, "PlaidMLExecutableNetwork::CreateInferRequestImpl>");
  IVLOG(2, "networkInputs: " << networkInputs);
  IVLOG(2, "networkOutputs: " << networkOutputs);
  IVLOG(3, "tensorIONameMap_: " << tensorIONameMap_);
  std::vector<edsl::Tensor> inputs;
  for (const auto& kvp : networkInputs) {
    IVLOG(2, "input: " << kvp.first);
    inputs.push_back(tensorIONameMap_.at(kvp.first));
  }
  std::vector<edsl::Tensor> outputs;
  for (const auto& kvp : networkOutputs) {
    IVLOG(2, "output: " << kvp.first);
    outputs.push_back(tensorIONameMap_.at(kvp.first));
  }
  Program program = edsl::buildProgram("ie", inputs, outputs);
  program.compile();
  return std::make_shared<PlaidMLInferRequest>(networkInputs, networkOutputs, program);
}

PlaidMLExecutableNetwork::PlaidMLExecutableNetwork(const ICNNNetwork& network, const std::string& device) {
  auto fcn = network.getFunction();
  IE_ASSERT(fcn);  // PlaidML requires that the nGraph-based API be used
  IVLOG(2, "Layers:");
  for (const std::shared_ptr<ngraph::Node>& node : fcn->get_ordered_ops()) {
    IVLOG(2, "  " << node->description() << ": " << node->get_name() << "... " << node->get_friendly_name());
    if (ngraph::op::is_constant(node)) {
      handleConstant(node);
    } else if (ngraph::op::is_parameter(node)) {
      handleParameter(node);
    } else if (ngraph::op::is_output(node)) {
      handleOutput(node);
    } else {
      handleOp(node);
    }
  }
}

void PlaidMLExecutableNetwork::handleConstant(const std::shared_ptr<ngraph::Node>& node) {
  IE_ASSERT(node->get_output_size() == 1);
  IE_ASSERT(node->description() == "Constant");
  auto type = to_plaidml(node->get_element_type());
  std::vector<int64_t> dims{node->get_shape().begin(), node->get_shape().end()};
  TensorShape shape(type, dims);
  Buffer buffer(shape);
  // Specially resolve the constant-creating op
  Context ctx{node.get()};
  auto* layer = dynamic_cast<ngraph::opset1::Constant*>(ctx.layer);
  buffer.copy_from(layer->get_data_ptr());
  auto tensor = edsl::Constant(buffer, node->get_friendly_name());
  IVLOG(3, "    Adding constant named '" << node->get_output_tensor_name(0) << "'");
  tensorMap_[node->output(0).get_tensor_ptr()] = tensor;
}

void PlaidMLExecutableNetwork::handleParameter(const std::shared_ptr<ngraph::Node>& node) {
  IE_ASSERT(node->get_output_size() == 1);
  std::vector<int64_t> dims{node->get_shape().begin(), node->get_shape().end()};
  auto type = to_plaidml(node->get_element_type());
  auto tensor = edsl::Placeholder(type, dims, node->get_friendly_name());
  IVLOG(3, "    Adding parameter named '" << node->get_name() << "'");
  tensorMap_[node->output(0).get_tensor_ptr()] = tensor;
  tensorIONameMap_[node->get_name()] = tensor;
}

void PlaidMLExecutableNetwork::handleOutput(const std::shared_ptr<ngraph::Node>& node) {
  // The OV output name is the name of the node _prior_ to the result
  tensorIONameMap_[node->inputs()[0].get_source_output().get_node()->get_name()] =
      tensorMap_.at(node->input(0).get_tensor_ptr());
}

void PlaidMLExecutableNetwork::handleOp(const std::shared_ptr<ngraph::Node>& node) {
  auto op = OpsRegistry::instance()->resolve(node->description());
  if (!op) {
    THROW_IE_EXCEPTION << "Unsupported operation: " << node->description();
  }

  Context ctx{node.get()};
  for (const auto& input : node->inputs()) {
    if (VLOG_IS_ON(1)) {
      const auto& src_output = input.get_source_output();
      const auto& name = src_output.get_node()->get_output_tensor_name(src_output.get_index());
      IVLOG(1, "    input: " << name);
    }
    auto tensor = tensorMap_.at(input.get_tensor_ptr());
    ctx.operands.push_back(tensor);
  }
  auto value = op(ctx);
  auto tuple = value.as_tuple();
  IE_ASSERT(tuple.size() == node->get_output_size());
  for (unsigned i = 0; i < tuple.size(); i++) {
    auto tensor = tuple.at(i).as_tensor();
    if (VLOG_IS_ON(1)) {
      const auto& name = node->get_output_tensor_name(i);
      IVLOG(1, "    output: " << name);
    }
    tensorMap_[node->output(i).get_tensor_ptr()] = tensor;
  }
}

void PlaidMLExecutableNetwork::GetMetric(const std::string& name, Parameter& result, ResponseDesc* resp) const {
  IVLOG(1, "PlaidMLExecutableNetwork::GetMetric> " << name);
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
