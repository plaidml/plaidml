// Copyright 2020 Intel Corporation

#include <iostream>

#include "networks/oplib/oplib.h"
#include "plaidml/exec/exec.h"
#include "plaidml/op/op.h"

// TODO: command line parsing to dispatch on network to run
int main(int argc, char** argv) {
  try {
    plaidml::init();
    plaidml::edsl::init();
    plaidml::op::init();
    plaidml::exec::init();
    auto program = networks::oplib::buildResnet50();
    std::cout << program.str() << std::endl;
    program.compile();
    auto exe = plaidml::exec::Executable(program);
    std::cout << "Running..." << std::endl;
    std::vector<plaidml::Buffer> inputs;
    for (const plaidml::TensorShape& shape : program.inputs()) {
      inputs.emplace_back(shape);
    }
    std::vector<plaidml::Buffer> outputs;
    for (const plaidml::TensorShape& shape : program.outputs()) {
      outputs.emplace_back(shape);
    }
#if !defined(_WIN32)
    exe.run(inputs, outputs);
#endif
    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "Caught unhandled exception: " << ex.what() << std::endl;
    return EXIT_FAILURE;
  } catch (...) {
    std::cerr << "Caught unhandled exception" << std::endl;
    return EXIT_FAILURE;
  }
}
