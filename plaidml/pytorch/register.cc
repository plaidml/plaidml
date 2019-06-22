// Copyright 2019, Intel Corporation.

#include <pybind11/pybind11.h>
#include <torch/csrc/autograd/record_function.h>
#include <torch/csrc/jit/custom_operator.h>
#include <torch/csrc/jit/pass_manager.h>
#include <torch/csrc/jit/passes/graph_fuser.h>

#include "plaidml/pytorch/compiler.h"
#include "plaidml/pytorch/logging.h"

// Based largely on https://github.com/pytorch/tvm
// https://jott.live/markdown/Writing%20a%20Toy%20Backend%20Compiler%20for%20PyTorch
// https://github.com/pytorch/pytorch/issues/19092
// http://blog.christianperone.com/2018/03/pytorch-internal-architecture-tour
//
// Linux currently suffers from this:
//   undefined symbol: _ZN3c106Symbol14fromQualStringERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
// This appears to be due to:
//   https://davmac.wordpress.com/2015/07/19/tale-of-two-abis/
//   https://developers.redhat.com/blog/2015/02/05/gcc5-and-the-c11-abi/
// One solution is to compile the world with: `-D_GLIBCXX_USE_CXX11_ABI=0`

using namespace torch::jit;  // NOLINT

static bool g_fusion_enabled = false;

size_t g_verbosity = 0;

PYBIND11_MODULE(plaidml_pytorch, module) {
  auto ctx = std::make_shared<vertexai::ctx>();
  const auto& devices = vertexai::plaidml::enumerate_devices(ctx);
  if (devices.empty()) {
    throw std::runtime_error("No available devices.");
  }
  auto device = devices.at(0).open();

  RegisterPass pass([](std::shared_ptr<Graph> graph) {
    if (g_fusion_enabled) {
      CustomFuseGraph(graph, Compiler::is_supported, Compiler::symbol);
    }
  });

  auto options = c10::OperatorOptions();
  options.setAliasAnalysis(AliasAnalysisKind::PURE);

  RegisterOperators op({Operator(
      Compiler::symbol,
      [device](const Node* node) {
        auto compiler = std::make_shared<Compiler>(device, node);
        return [compiler](Stack& stack) {
          RECORD_FUNCTION("PlaidML", std::vector<c10::IValue>());
          compiler->run(&stack);
          return 0;
        };
      },
      options)});

  module.def("enable", []() { g_fusion_enabled = true; });
  module.def("disable", []() { g_fusion_enabled = false; });
  module.def("set_vlog", [](size_t verbosity) { g_verbosity = verbosity; });
}
