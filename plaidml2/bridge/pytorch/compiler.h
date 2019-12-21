// Copyright 2019, Intel Corporation.

#pragma once

#include <torch/csrc/jit/argument_spec.h>
#include <torch/csrc/jit/ir.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "plaidml2/edsl/edsl.h"
#include "plaidml2/exec/exec.h"

class Executable {
 public:
  Executable(const std::string& device_id,                       //
             const std::string& target_id,                       //
             const std::vector<plaidml2::edsl::Tensor>& inputs,  //
             const std::vector<plaidml2::edsl::Tensor>& outputs);

  at::ArrayRef<torch::jit::IValue> run(at::ArrayRef<torch::jit::IValue> inputs);

 private:
  std::string device_id_;
  std::string target_id_;
  std::unique_ptr<plaidml2::edsl::Program> program_;
  std::unique_ptr<plaidml2::exec::Binder> binder_;
  std::shared_ptr<plaidml2::exec::Executable> exec_;
  // std::vector<plaidml2::exec::Binding> input_bindings_;
  // std::vector<plaidml2::exec::Binding> output_bindings_;
  std::vector<torch::jit::IValue> output_ivalues_;
  std::string name_;
};

class Compiler {
 public:
  explicit Compiler(const std::string& device_id,  //
                    const std::string& target_id,  //
                    const torch::jit::Node* node);

  void run(torch::jit::Stack* stack);

  static const at::Symbol symbol;
  static bool is_supported(torch::jit::Node* node);

 private:
  std::shared_ptr<Executable> compile(at::ArrayRef<torch::jit::IValue> inputs);

 private:
  std::string device_id_;
  std::string target_id_;
  std::shared_ptr<torch::jit::Graph> subgraph_;
  std::unordered_map<torch::jit::CompleteArgumentSpec, std::shared_ptr<Executable>> cache_;
};
