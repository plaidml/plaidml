// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ie_layers.h>

#include "plaidml_infer_request.hpp"
#include "plaidml_util.hpp"

using namespace InferenceEngine;
using namespace PlaidMLPlugin;

PlaidMLInferRequest::PlaidMLInferRequest(InferenceEngine::InputsDataMap networkInputs,
                                         InferenceEngine::OutputsDataMap networkOutputs,
                                         const std::shared_ptr<State>& state)
    : InferRequestInternal(networkInputs, networkOutputs), state_(state) {
  std::vector<plaidml::edsl::Tensor> outputs;

  for (const auto& out : networkOutputs) {
    outputs.push_back(state_->slot<plaidml::edsl::Tensor>()[out.first]);
  }

  // binder_.reset(new plaidml::exec::Binder(plaidml::edsl::Program("", outputs)));

  // PML get device name and target name from enviroment variable for the Binder::compile
  // but device name passing as function parameter for creating plaidml::Buffer.
  // Target name remains empty if environment variable not set. It will cause an exception.
  // So we need to set device name and target name manually that avoid problems
  binder_->set_device(state_->device());
  // binder_->set_target(state_->target());

  auto all_weights = state_->slot<std::vector<plaidml::exec::Binding>>();
  for (const auto& layer_weights : all_weights) {
    for (const auto& weight : layer_weights.second) {
      binder_->set_input(weight.tensor, weight.buffer);
    }
  }

  AllocateInputs();
  AllocateOutputs();

  exec_ = binder_->compile();
}

void PlaidMLInferRequest::InferImpl() {
  execDataPreprocessing(_inputs);

  SyncInput();
  exec_->run();
  SyncOutput();
}

void PlaidMLInferRequest::GetPerformanceCounts(
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& perfMap) const {
  throw std::logic_error("PlaidMLInferRequest::GetPerformanceCounts not implemented");
}

void PlaidMLInferRequest::AllocateInputs() {
  for (const auto& p : _networkInputs) {
    const auto& desc = p.second->getTensorDesc();
    const auto& name = p.first;
    const auto& tensor = state_->slot<plaidml::edsl::Tensor>()[name];

    // binder_->set_input(tensor, plaidml::Buffer(state_->device(), {tensor.shape().dtype(),
    // tensor.shape().int_dims()}));

    auto insert_info = _inputs.emplace(name, util::make_shared_blob(desc));
    insert_info.first->second->allocate();
  }
}

void PlaidMLInferRequest::AllocateOutputs() {
  for (const auto& p : _networkOutputs) {
    const auto& desc = p.second->getTensorDesc();
    const auto& name = p.first;

    const auto& tensor = state_->slot<plaidml::edsl::Tensor>()[name];
    // binder_->set_output(tensor, plaidml::Buffer(state_->device(), {tensor.shape().dtype(),
    // tensor.shape().int_dims()}));

    auto insert_info = _outputs.emplace(name, util::make_shared_blob(desc));
    insert_info.first->second->allocate();
  }
}

void PlaidMLInferRequest::SyncInput() {
  for (const auto& p : _networkInputs) {
    const auto& name = p.first;
    const auto& tensor = state_->slot<plaidml::edsl::Tensor>()[name];
    auto& in = _inputs[name];
    const auto& desc = in->getTensorDesc();
    const auto& dims = desc.getDims();

    if (desc.getLayout() == Layout::NCHW) {
      auto in_buffer = binder_->input(tensor);
      util::transpose(in->buffer().as<uint8_t*>(), dims, {0, 2, 3, 1},
                      reinterpret_cast<uint8_t*>(in_buffer.mmap_discard().data()), in->element_size());
    } else {
      binder_->input(tensor).copy_from(_inputs[name]->buffer());
    }
  }
}

void PlaidMLInferRequest::SyncOutput() {
  for (const auto& p : _networkOutputs) {
    const auto& name = p.first;
    const auto& tensor = state_->slot<plaidml::edsl::Tensor>()[name];
    auto& out = _outputs[name];
    const auto& desc = out->getTensorDesc();
    const auto& dims = desc.getDims();

    if (desc.getLayout() == Layout::NCHW) {
      auto out_buffer = binder_->output(tensor);
      util::transpose(reinterpret_cast<uint8_t*>(out_buffer.mmap_current().data()),
                      {dims[0], dims[2], dims[3], dims[1]}, {0, 3, 1, 2}, out->buffer().as<uint8_t*>(),
                      out->element_size());
    } else {
      binder_->output(tensor).copy_into(_outputs[name]->buffer());
    }
  }
}
