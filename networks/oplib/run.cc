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
    auto exe = networks::oplib::createDefaultExecutable(program);
    std::cout << "Running..." << std::endl;
#if !defined(_WIN32)
    exe.run();
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
