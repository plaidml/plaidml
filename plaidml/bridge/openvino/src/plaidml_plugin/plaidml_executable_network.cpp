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
  std::vector<plaidml::edsl::Tensor> outputs;
  IVLOG(2, "networkOutputs: " << networkOutputs);
  IVLOG(3, "tensorIOMap_: " << tensorIOMap_);
  for (const auto& kvp : networkOutputs) {
    IVLOG(2, "output: " << kvp.first);
    outputs.push_back(tensorIOMap_.at(kvp.first));
  }
  auto program = edsl::ProgramBuilder("ie", outputs).compile();
  return std::make_shared<PlaidMLInferRequest>(networkInputs, networkOutputs, program, tensorIOMap_);
}

PlaidMLExecutableNetwork::PlaidMLExecutableNetwork(const ICNNNetwork& network, const std::string& device) {
  InputsDataMap inputMap;
  auto fcn = network.getFunction();
  IE_ASSERT(fcn);  // PlaidML requires that the nGraph-based API be used
  IVLOG(2, "Layers:");
  for (auto& node : fcn->get_ordered_ops()) {
    IVLOG(2, "  " << node->description() << ": " << node->get_name() << "... " << node->get_friendly_name());
    if (ngraph::op::is_constant(node)) {
      IE_ASSERT(node->get_output_size() == 1);
      IE_ASSERT(node->description() == "Constant");
      auto type = to_plaidml(node->get_element_type());
      std::vector<int64_t> dims{node->get_shape().begin(), node->get_shape().end()};
      TensorShape ts(type, dims);
      Buffer buffer(device, ts);
      // Specially resolve the constant-creating op
      Context ctx{node.get()};
      auto* layer = dynamic_cast<ngraph::opset1::Constant*>(ctx.layer);
      buffer.copy_from(layer->get_data_ptr());
      auto tensor = edsl::Constant(type, buffer, dims, node->get_friendly_name());
      IVLOG(3, "    Adding constant named '" << node->get_output_tensor_name(0) << "'");
      tensorMap_[node->get_output_tensor_name(0)] = tensor;
      continue;
    } else if (ngraph::op::is_parameter(node)) {
      IE_ASSERT(node->get_output_size() == 1);
      std::vector<int64_t> dims{node->get_shape().begin(), node->get_shape().end()};
      auto type = to_plaidml(node->get_element_type());
      auto tensor = edsl::Placeholder(edsl::LogicalShape(type, dims), node->get_friendly_name());
      IVLOG(3, "    Adding placeholder named '" << node->get_output_tensor_name(0) << "'");
      tensorMap_[node->get_output_tensor_name(0)] = tensor;
      IVLOG(3, "    Also, aliasing " << node->get_output_tensor_name(0) << " as " << node->get_friendly_name());
      tensorIOMap_[node->get_friendly_name()] = tensor;
      continue;
    } else if (ngraph::op::is_output(node)) {
      const auto& src_output = node->inputs()[0].get_source_output();
      const auto& friendly_name = src_output.get_node()->get_friendly_name();
      const auto& original_name = src_output.get_node()->get_output_tensor_name(src_output.get_index());
      IVLOG(3, "At an output node, aliasing " << original_name << " as " << friendly_name);
      tensorIOMap_[friendly_name] = tensorMap_.at(original_name);
      continue;
    }

    auto op = OpsRegistry::instance()->resolve(node->description());
    if (!op) {
      THROW_IE_EXCEPTION << "Unsupported operation: " << node->description();
    }

    Context ctx{node.get()};
    for (const auto& input : node->inputs()) {
      const auto& src_output = input.get_source_output();
      const auto& name = src_output.get_node()->get_output_tensor_name(src_output.get_index());
      IVLOG(1, "    input: " << name);
      auto tensor = tensorMap_.at(name);
      ctx.operands.push_back(tensor);
    }
    auto value = op(ctx);
    auto tuple = value.as_tuple();
    IE_ASSERT(tuple.size() == node->get_output_size());
    for (unsigned i = 0; i < tuple.size(); i++) {
      auto tensor = tuple.at(i).as_tensor();
      const auto& name = node->get_output_tensor_name(i);
      IVLOG(1, "    output: " << name);
      tensorMap_[name] = tensor;
    }
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
