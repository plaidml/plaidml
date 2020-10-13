// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_infer_request.hpp"

#include "ie_layers.h"  // NOLINT[build/include_subdir]

#include "pmlc/util/logging.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]

static Blob::Ptr make_shared_blob(const TensorDesc& desc) {
  const auto prec = desc.getPrecision();

#define CASE(prec) \
  case prec:       \
    return InferenceEngine::make_shared_blob<PrecisionTrait<prec>::value_type>(desc);

  switch (prec) {
    CASE(Precision::FP32);
    CASE(Precision::FP16);
    CASE(Precision::Q78);
    CASE(Precision::U16);
    CASE(Precision::U8);
    CASE(Precision::I8);
    CASE(Precision::BOOL);
    CASE(Precision::I16);
    CASE(Precision::I32);
    CASE(Precision::I64);
    CASE(Precision::BIN);

    default:
      THROW_IE_EXCEPTION << "The plugin does not support input " << prec.name() << " precision";
  }
}

namespace PlaidMLPlugin {

PlaidMLInferRequest::PlaidMLInferRequest(const InputsDataMap& networkInputs, const OutputsDataMap& networkOutputs,
                                         const Program& program)
    : InferRequestInternal(networkInputs, networkOutputs), program_(program) {
  IVLOG(1, "Program:\n" << program.str());
  AllocateInputs();
  AllocateOutputs();
  exec_ = std::make_shared<exec::Executable>(program);
}

void PlaidMLInferRequest::InferImpl() {
  IVLOG(1, "PlaidMLInferRequest::InferImpl>");
  IVLOG(2, "  _inputs: " << _inputs);
  execDataPreprocessing(_inputs);

  SyncInput();
  exec_->run(input_buffers_, output_buffers_);
  SyncOutput();
}

void PlaidMLInferRequest::GetPerformanceCounts(std::map<std::string, InferenceEngineProfileInfo>& perfMap) const {
  throw std::logic_error("PlaidMLInferRequest::GetPerformanceCounts not implemented");
}

void PlaidMLInferRequest::AllocateInputs() {
  size_t i = 0;
  auto inputs = program_.inputs();
  for (const auto& kvp : _networkInputs) {
    const auto& name = kvp.first;
    const auto& desc = kvp.second->getTensorDesc();
    auto info = _inputs.emplace(name, make_shared_blob(desc));
    info.first->second->allocate();
    input_buffers_.emplace_back(inputs[i++]);
  }
}

void PlaidMLInferRequest::AllocateOutputs() {
  size_t i = 0;
  auto outputs = program_.outputs();
  for (const auto& kvp : _networkOutputs) {
    const auto& name = kvp.first;
    const auto& desc = kvp.second->getTensorDesc();
    auto info = _outputs.emplace(name, make_shared_blob(desc));
    info.first->second->allocate();
    output_buffers_.emplace_back(outputs[i++]);
  }
}

void PlaidMLInferRequest::SyncInput() {
  size_t i = 0;
  for (const auto& kvp : _networkInputs) {
    const auto& name = kvp.first;
    input_buffers_[i++].copy_from(_inputs[name]->buffer());
  }
}

void PlaidMLInferRequest::SyncOutput() {
  size_t i = 0;
  for (const auto& kvp : _networkOutputs) {
    const auto& name = kvp.first;
    output_buffers_[i++].copy_into(_outputs[name]->buffer());
  }
}

}  // namespace PlaidMLPlugin
