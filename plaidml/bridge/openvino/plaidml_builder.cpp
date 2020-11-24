// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_builder.hpp"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ngraph/function.hpp"
#include "ngraph/opsets/opset1.hpp"

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"

namespace PlaidMLPlugin {

namespace {

class ProgramBuilder {
 public:
  explicit ProgramBuilder(const InferenceEngine::ICNNNetwork& network);

  plaidml::Program build();

 private:
  void handleConstant(const std::shared_ptr<ngraph::Node>& node);
  void handleParameter(const std::shared_ptr<ngraph::Node>& node);
  void handleOutput(const std::shared_ptr<ngraph::Node>& node);
  void handleOp(const std::shared_ptr<ngraph::Node>& node);

  const InferenceEngine::ICNNNetwork& network;

  // Lets us look up the PlaidML tensor by the name of the node that produces it and the index of which output it is
  std::map<std::pair<std::string, size_t>, plaidml::edsl::Tensor> tensorMap;

  // Go from the names OV uses for a networks inputs and outputs to the corresponding PlaidML Tensor
  std::map<std::string, plaidml::edsl::Tensor> tensorIONameMap;
};

ProgramBuilder::ProgramBuilder(const InferenceEngine::ICNNNetwork& network) : network(network) {}

plaidml::Program ProgramBuilder::build() {
  std::shared_ptr<const ngraph::Function> func = network.getFunction();
  IE_ASSERT(func);  // PlaidML requires that the nGraph-based API be used
  for (const std::shared_ptr<ngraph::Node>& node : func->get_ordered_ops()) {
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

  InferenceEngine::InputsDataMap networkInputs;
  network.getInputsInfo(networkInputs);
  std::vector<plaidml::edsl::Tensor> inputs;
  for (const auto& kvp : networkInputs) {
    inputs.push_back(tensorIONameMap.at(kvp.first));
  }

  InferenceEngine::OutputsDataMap networkOutputs;
  network.getOutputsInfo(networkOutputs);
  std::vector<plaidml::edsl::Tensor> outputs;
  for (const auto& kvp : networkOutputs) {
    outputs.push_back(tensorIONameMap.at(kvp.first));
  }

  return plaidml::edsl::buildProgram("ie", inputs, outputs);
}

void ProgramBuilder::handleConstant(const std::shared_ptr<ngraph::Node>& node) {
  IE_ASSERT(node->get_output_size() == 1);
  IE_ASSERT(node->description() == "Constant");
  plaidml::DType type = to_plaidml(node->get_element_type());
  std::vector<int64_t> dims{node->get_shape().begin(), node->get_shape().end()};
  plaidml::TensorShape shape(type, dims);
  plaidml::Buffer buffer(shape);
  Context ctx{node.get()};
  auto* layer = dynamic_cast<ngraph::opset1::Constant*>(ctx.layer);
  buffer.copy_from(layer->get_data_ptr());
  plaidml::edsl::Tensor tensor = plaidml::edsl::Constant(buffer, node->get_friendly_name());
  tensorMap[std::make_pair(node->get_name(), 0)] = tensor;
}

void ProgramBuilder::handleParameter(const std::shared_ptr<ngraph::Node>& node) {
  IE_ASSERT(node->get_output_size() == 1);
  std::vector<int64_t> dims{node->get_shape().begin(), node->get_shape().end()};
  plaidml::DType type = to_plaidml(node->get_element_type());
  plaidml::edsl::Tensor tensor = plaidml::edsl::Placeholder(type, dims, node->get_friendly_name());
  tensorMap[std::make_pair(node->get_name(), 0)] = tensor;
  tensorIONameMap[node->get_friendly_name()] = tensor;
}

void ProgramBuilder::handleOutput(const std::shared_ptr<ngraph::Node>& node) {
  // The OV output name is the name of the node _prior_ to the result
  // When there are multiple outputs, it has .# appended, where # is the output index
  const ngraph::Output<ngraph::Node>& src_output = node->input(0).get_source_output();
  const ngraph::Node* src_node = src_output.get_node();
  std::string name = src_node->get_friendly_name();
  if (src_node->get_output_size() > 1) {
    name += "." + std::to_string(src_output.get_index());
  }
  tensorIONameMap[name] = tensorMap.at(std::make_pair(src_node->get_name(), src_output.get_index()));
}

struct PlaidMLAttributeVisitor : public ngraph::AttributeVisitor {
  plaidml::edsl::Dictionary attrs;

  void on_adapter(const std::string& name, ngraph::ValueAccessor<void>& adapter) final {
    THROW_IE_EXCEPTION << "Unsupported 'void' attribute: " << name;
  }

  void on_adapter(const std::string& name, ngraph::ValueAccessor<std::string>& adapter) final {}

  void on_adapter(const std::string& name, ngraph::ValueAccessor<bool>& adapter) final {
    attrs[name] = plaidml::edsl::Value(adapter.get());
  }

  void on_adapter(const std::string& name, ngraph::ValueAccessor<int64_t>& adapter) final {
    attrs[name] = plaidml::edsl::Value(adapter.get());
  }

  void on_adapter(const std::string& name, ngraph::ValueAccessor<double>& adapter) final {
    attrs[name] = plaidml::edsl::Value(adapter.get());
  }

  void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<std::string>>& adapter) final {
    attrs[name] = plaidml::edsl::make_tuple(adapter.get());
  }

  void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<float>>& adapter) final {
    attrs[name] = plaidml::edsl::make_tuple(adapter.get());
  }

  void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<double>>& adapter) final {
    attrs[name] = plaidml::edsl::make_tuple(adapter.get());
  }

  void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int8_t>>& adapter) final {
    attrs[name] = plaidml::edsl::make_tuple(adapter.get());
  }

  void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int16_t>>& adapter) final {
    attrs[name] = plaidml::edsl::make_tuple(adapter.get());
  }

  void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int32_t>>& adapter) final {
    attrs[name] = plaidml::edsl::make_tuple(adapter.get());
  }

  void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<int64_t>>& adapter) final {
    attrs[name] = plaidml::edsl::make_tuple(adapter.get());
  }

  void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<uint8_t>>& adapter) final {
    attrs[name] = plaidml::edsl::make_tuple(adapter.get());
  }

  void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<uint16_t>>& adapter) final {
    attrs[name] = plaidml::edsl::make_tuple(adapter.get());
  }

  void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<uint32_t>>& adapter) final {
    attrs[name] = plaidml::edsl::make_tuple(adapter.get());
  }

  void on_adapter(const std::string& name, ngraph::ValueAccessor<std::vector<uint64_t>>& adapter) final {
    attrs[name] = plaidml::edsl::make_tuple(adapter.get());
  }

  void on_adapter(const std::string& name, ngraph::ValueAccessor<void*>& adapter) final {
    THROW_IE_EXCEPTION << "Unsupported 'void*' attribute: " << name;
  }
};

void ProgramBuilder::handleOp(const std::shared_ptr<ngraph::Node>& node) {
  const Op op = OpsRegistry::instance()->resolve(node->description());
  if (!op) {
    THROW_IE_EXCEPTION << "Unsupported operation: " << node->description();
  }

  Context ctx{node.get()};
  for (const auto& input : node->inputs()) {
    const ngraph::Output<ngraph::Node>& src_output = input.get_source_output();
    const std::string& name = src_output.get_node()->get_name();
    size_t index = src_output.get_index();
    plaidml::edsl::Tensor tensor = tensorMap.at(std::make_pair(name, index));
    ctx.operands.push_back(tensor);
  }
  PlaidMLAttributeVisitor visitor;
  node->visit_attributes(visitor);
  plaidml::edsl::TensorVec tuple = plaidml::edsl::layer("ng." + node->description(), visitor.attrs, [&]() {
    plaidml::edsl::Value value = op(ctx);
    std::vector<plaidml::edsl::Value> tuple = value.as_tuple();
    plaidml::edsl::TensorVec outputs;
    outputs.reserve(tuple.size());
    for (plaidml::edsl::Value output : tuple) {
      outputs.push_back(output.as_tensor());
    }
    return outputs;
  });
  IE_ASSERT(tuple.size() == node->get_output_size());
  for (unsigned i = 0; i < tuple.size(); i++) {
    plaidml::edsl::Tensor tensor = tuple.at(i);
    tensorMap[std::make_pair(node->get_name(), i)] = tensor;
  }
}

}  // namespace

plaidml::Program buildProgram(const InferenceEngine::ICNNNetwork& network) { return ProgramBuilder(network).build(); }

}  // namespace PlaidMLPlugin
