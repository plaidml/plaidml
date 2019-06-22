// Copyright 2019, Intel Corporation.

#pragma once

#include <torch/csrc/jit/argument_spec.h>
#include <torch/csrc/jit/ir.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "plaidml/edsl/edsl.h"
#include "plaidml/plaidml++.h"

class Executable {
 public:
  Executable(const vertexai::plaidml::device& device,                     //
             const std::vector<vertexai::plaidml::edsl::Tensor>& inputs,  //
             const std::vector<vertexai::plaidml::edsl::Tensor>& outputs);

  std::vector<torch::jit::IValue> run(at::ArrayRef<torch::jit::IValue>* inputs);

 private:
  vertexai::plaidml::device device_;
  std::unique_ptr<vertexai::plaidml::edsl::Program> program_;
  vertexai::plaidml::executable exec_;
  std::vector<vertexai::plaidml::binding> input_bindings_;
  std::vector<vertexai::plaidml::binding> output_bindings_;
  std::vector<torch::jit::IValue> output_ivalues_;
  std::shared_ptr<vertexai::ctx> ctx_;
  std::string name_;
};

class Compiler {
 public:
  explicit Compiler(const vertexai::plaidml::device& device, const torch::jit::Node* node);

  void run(torch::jit::Stack* stack);

  static const at::Symbol symbol;
  static bool is_supported(torch::jit::Node* node);

 private:
  std::shared_ptr<Executable> compile(at::ArrayRef<torch::jit::IValue>* inputs);

 private:
  vertexai::plaidml::device device_;
  std::shared_ptr<torch::jit::Graph> subgraph_;
  std::unordered_map<torch::jit::CompleteArgumentSpec, std::shared_ptr<Executable>> cache_;
};
