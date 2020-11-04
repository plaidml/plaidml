// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_program_builder.hpp"

#include <memory>
#include <utility>
#include <vector>

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

using namespace plaidml;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

PlaidMLProgramBuilder::PlaidMLProgramBuilder(const InferenceEngine::ICNNNetwork& network)
    : fcn_(network.getFunction()) {
  IE_ASSERT(fcn_);  // PlaidML requires that the nGraph-based API be used
  network.getInputsInfo(networkInputs_);
  network.getOutputsInfo(networkOutputs_);
}

Program PlaidMLProgramBuilder::program() {
  if (!graph_parsed) {
    for (const std::shared_ptr<ngraph::Node>& node : fcn_->get_ordered_ops()) {
      if (node->description() == "Constant") {
        handleConstant(node);
      } else if (node->description() == "Parameter") {
        handleParameter(node);
      } else if (node->description() == "Result") {
        handleOutput(node);
      } else {
        handleOp(node);
      }
    }
    graph_parsed = true;
  }

  std::vector<edsl::Tensor> inputs;
  for (const auto& kvp : networkInputs_) {
    inputs.push_back(tensorIONameMap_.at(kvp.first));
  }
  std::vector<edsl::Tensor> outputs;
  for (const auto& kvp : networkOutputs_) {
    outputs.push_back(tensorIONameMap_.at(kvp.first));
  }
  return edsl::buildProgram("ie", inputs, outputs);
}

void PlaidMLProgramBuilder::handleConstant(const std::shared_ptr<ngraph::Node>& node) {
  IE_ASSERT(node->get_output_size() == 1);
  IE_ASSERT(node->description() == "Constant");
  auto type = to_plaidml(node->get_element_type());
  std::vector<int64_t> dims{node->get_shape().begin(), node->get_shape().end()};
  TensorShape shape(type, dims);
  Buffer buffer(shape);
  Context ctx{node.get()};
  auto* layer = dynamic_cast<ngraph::opset1::Constant*>(ctx.layer);
  buffer.copy_from(layer->get_data_ptr());
  auto tensor = edsl::Constant(buffer, node->get_friendly_name());
  tensorMap_[std::make_pair(node->get_name(), 0)] = tensor;
}

void PlaidMLProgramBuilder::handleParameter(const std::shared_ptr<ngraph::Node>& node) {
  IE_ASSERT(node->get_output_size() == 1);
  std::vector<int64_t> dims{node->get_shape().begin(), node->get_shape().end()};
  auto type = to_plaidml(node->get_element_type());
  auto tensor = edsl::Placeholder(type, dims, node->get_friendly_name());
  tensorMap_[std::make_pair(node->get_name(), 0)] = tensor;
  tensorIONameMap_[node->get_friendly_name()] = tensor;
}

void PlaidMLProgramBuilder::handleOutput(const std::shared_ptr<ngraph::Node>& node) {
  // The OV output name is the name of the node _prior_ to the result
  // When there are multiple outputs, it has .# appended, where # is the output index
  const auto& src_output = node->input(0).get_source_output();
  const auto& src_node = src_output.get_node();
  std::string name = src_node->get_friendly_name();
  if (src_node->get_output_size() > 1) {
    name += "." + std::to_string(src_output.get_index());
  }
  tensorIONameMap_[name] = tensorMap_.at(std::make_pair(src_node->get_name(), src_output.get_index()));
}

void PlaidMLProgramBuilder::handleOp(const std::shared_ptr<ngraph::Node>& node) {
  auto op = OpsRegistry::instance()->resolve(node->description());
  if (!op) {
    THROW_IE_EXCEPTION << "Unsupported operation: " << node->description();
  }

  Context ctx{node.get()};
  for (const auto& input : node->inputs()) {
    const auto& src_output = input.get_source_output();
    const auto& name = src_output.get_node()->get_name();
    const auto& index = src_output.get_index();
    auto tensor = tensorMap_.at(std::make_pair(name, index));
    ctx.operands.push_back(tensor);
  }
  auto value = op(ctx);
  auto tuple = value.as_tuple();
  IE_ASSERT(tuple.size() == node->get_output_size());
  for (unsigned i = 0; i < tuple.size(); i++) {
    auto tensor = tuple.at(i).as_tensor();
    tensorMap_[std::make_pair(node->get_name(), i)] = tensor;
  }
}

Program makePlaidMLProgram(const InferenceEngine::ICNNNetwork& network) {
  return PlaidMLProgramBuilder(network).program();
}

}  // namespace PlaidMLPlugin
