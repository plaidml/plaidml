// Copyright 2021 Intel Corporation

#include <iostream>
#include <string>

#include "llvm/Support/FormatVariadic.h"

#include "plaidml/exec/exec.h"
#include "plaidml/op/op.h"

using namespace plaidml;  // NOLINT

Program buildLayer(int64_t N, int64_t K, int64_t C, int64_t H, int64_t W, int64_t R, int64_t S, int64_t stride) {
  edsl::Tensor I = edsl::Placeholder(DType::FLOAT32, {N, H, W, C});
  edsl::Tensor F = edsl::Placeholder(DType::FLOAT32, {R, S, C, K});
  std::string name = llvm::formatv("conv2d:{0}:{1}:{2}:{3}:{4}:{5}:{6}:{7}", N, K, C, H, W, R, S, stride);
  edsl::Tensor T1 = edsl::trace(I, name);
  edsl::Tensor O = op::convolution(T1, F)  //
                       .name("conv2d")
                       .strides({stride, stride})
                       .autopad_mode(op::AutoPadMode::VALID);
  edsl::Tensor T2 = edsl::trace(O, "done");
  return edsl::buildProgram("layer", {I, F}, {T2});
}

int main(int argc, char** argv) {
  try {
    plaidml::init();
    plaidml::edsl::init();
    plaidml::op::init();
    plaidml::exec::init();

    if (argc != 9) {
      std::cerr << "Usage: " << argv[0] << " N K C H W R S stride" << std::endl;
      std::cerr << "  Expected 9 arguments." << std::endl;
      return EXIT_FAILURE;
    }

    int64_t N = std::stoul(argv[1]);
    int64_t K = std::stoul(argv[2]);
    int64_t C = std::stoul(argv[3]);
    int64_t H = std::stoul(argv[4]);
    int64_t W = std::stoul(argv[5]);
    int64_t R = std::stoul(argv[6]);
    int64_t S = std::stoul(argv[7]);
    int64_t stride = std::stoul(argv[8]);

    Program program = buildLayer(N, K, C, H, W, R, S, stride);
    std::cout << program.str() << std::endl;
    program.compile();

    std::vector<Buffer> inputs;
    for (const TensorShape& shape : program.inputs()) {
      inputs.emplace_back(shape);
    }
    std::vector<Buffer> outputs;
    for (const TensorShape& shape : program.outputs()) {
      outputs.emplace_back(shape);
    }

    std::cout << "Running..." << std::endl;
    exec::Executable exe(program);
    for (int i = 0; i < 10; i++) {
      exe.run(inputs, outputs);
    }

    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "Caught unhandled exception: " << ex.what() << std::endl;
    return EXIT_FAILURE;
  } catch (...) {
    std::cerr << "Caught unhandled exception" << std::endl;
    return EXIT_FAILURE;
  }
}
