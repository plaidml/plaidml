// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// NB: IE_SET_METRIC requires this header to be included first
#include "ie_metric_helpers.hpp"

#include "plaidml_executable_network.hpp"

#include <vector>

#include <network_serializer.h>  // TODO: Sort/clean includes

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
  for (const auto& kvp : networkOutputs) {
    IVLOG(1, "output: " << kvp.first);
    outputs.push_back(tensorMap_.at(kvp.first));
  }
  auto program = edsl::ProgramBuilder("ie", outputs).compile();
  return std::make_shared<PlaidMLInferRequest>(networkInputs, networkOutputs, program, tensorMap_);
}

PlaidMLExecutableNetwork::PlaidMLExecutableNetwork(const ICNNNetwork& network, const std::string& device) {
  InputsDataMap inputMap;
  auto fcn = network.getFunction();
  // network.getInputsInfo(inputMap);

  IVLOG(1, "Layers:");
  for (auto& node : fcn->get_ordered_ops()) {
    IVLOG(1, "  " << node->description() << ": " << node->get_name());
    if (node->is_parameter()) {  // TODO: Verify this IDs inputs
      std::vector<int64_t> dims {node->get_shape().begin(), node->get_shape().end()};
      auto type = to_plaidml(node->get_element_type());
      auto tensor = edsl::Placeholder(edsl::LogicalShape(type, dims));
      tensorMap_[node->get_name()] = tensor;
      continue;
    } else if (node->is_constant()) { // TODO: Flip order with is_parameter
      // TODO
    }

    auto op = OpsRegistry::instance()->resolve(node->description());
    if (!op) {
      THROW_IE_EXCEPTION << "Unsupported operation: " << node->description();
    }

    Context ctx{node.get()};
    for (const auto& input : node->inputs()) {
      // TODO: n.b. each `input` is an instance of Input<Node>
      const auto& name = input.get_node()->get_name();  // TODO: Correct name?
      IVLOG(1, "    input: " << name);
      auto tensor = tensorMap_.at(name);
      ctx.operands.push_back(tensor);
    }
    auto value = op(ctx);
    auto tuple = value.as_tuple();
    IE_ASSERT(tuple.size() == node->get_output_size());
    for (unsigned i = 0; i < tuple.size(); i++) {
      auto tensor = tuple.at(i).as_tensor();
      const auto& name = node->output(i).get_node()->get_name();  // TODO: Correct name?
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
