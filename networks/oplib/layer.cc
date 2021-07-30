// Copyright 2020 Intel Corporation

#include <stdlib.h>

#include <iostream>

#include "networks/oplib/oplib.h"
#include "plaidml/exec/exec.h"
#include "plaidml/op/op.h"
#include "llvm/ADT/Twine.h"

plaidml::Program buildLayer(int64_t batch_size, int64_t output_filter_size, int64_t input_filter_size,
                            int64_t input_height, int64_t input_width, int64_t filter_height, int64_t filter_width,
                            int64_t stride) {
  auto I =
      plaidml::edsl::Placeholder(plaidml::DType::FLOAT32, {batch_size, input_height, input_width, input_filter_size});
  auto W = plaidml::edsl::Placeholder(plaidml::DType::FLOAT32,
                                      {filter_height, filter_width, input_filter_size, output_filter_size});

  auto T = plaidml::edsl::trace(
      I, "conv2d:" + llvm::Twine(I.compute_shape().sizes()[0]).str() + ":" +
             llvm::Twine(W.compute_shape().sizes()[3]).str() + ":" + llvm::Twine(I.compute_shape().sizes()[3]).str() +
             ":" + llvm::Twine(I.compute_shape().sizes()[1]).str() + ":" +
             llvm::Twine(I.compute_shape().sizes()[2]).str() + ":" + llvm::Twine(W.compute_shape().sizes()[0]).str() +
             ":" + llvm::Twine(W.compute_shape().sizes()[1]).str() + ":" + llvm::Twine((unsigned int)(stride)).str());

  plaidml::edsl::Tensor O = plaidml::op::convolution(T, W)
                                .name("conv2d")
                                .strides({stride, stride})
                                .autopad_mode(plaidml::op::AutoPadMode::VALID);

  auto done = plaidml::edsl::trace(O, "done\n");
  return plaidml::edsl::buildProgram("resnet50", {I, W}, {done});
}

// TODO: command line parsing to dispatch on network to run
int main(int argc, char** argv) {
  try {
    plaidml::init();
    plaidml::edsl::init();
    plaidml::op::init();
    plaidml::exec::init();

    if (argc != 9) {
      std::cerr << "Wrong number of arguments, should be 9" << std::endl;
      std::cerr << "Usage is ./layer batch ofm ifm height width r s stride" << std::endl;
      return EXIT_FAILURE;
    }

    int64_t N = (int64_t)atoi(argv[1]);
    int64_t K = (int64_t)atoi(argv[2]);
    int64_t C = (int64_t)atoi(argv[3]);
    int64_t H = (int64_t)atoi(argv[4]);
    int64_t W = (int64_t)atoi(argv[5]);
    int64_t R = (int64_t)atoi(argv[6]);
    int64_t S = (int64_t)atoi(argv[7]);
    int64_t stride = (int64_t)atoi(argv[8]);
    auto program = buildLayer(N, K, C, H, W, R, S, stride);  // networks::oplib::buildResnet50();
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
    for (int i = 0; i < 10; i++) exe.run(inputs, outputs);
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
