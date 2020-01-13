// Copyright 2019, Intel Corporation

#include "benchmark/benchmark.h"

#include "plaidml/exec/exec.h"
#include "plaidml/op/op.h"

namespace edsl = plaidml::edsl;
namespace exec = plaidml::exec;
namespace op = plaidml::op;

namespace networks::oplib {

namespace {

edsl::Tensor block(                      //
    const edsl::Tensor& I,               //
    const std::vector<edsl::Tensor>& W,  //
    const std::vector<edsl::Tensor>& B,  //
    const std::vector<int>& strides,     //
    bool use_shortcut_conv,              //
    const std::string& base_name) {
  // Note: The branch1 weights/biases are at the _end_ of the input vectors,
  // as their existence depends on use_shortcut_conv
  auto conv_2a = op::convolution(  //
      I,                           // input
      W[0],                        // weights
      strides,                     // strides
      {1, 1},                      // dilations
      {1, 1},                      // data dilations
      {},                          // filter shape (ie for transposed)
      1,                           // # of groups
      "valid",                     // autopadding
      {},                          // manual padding
      "nxc",                       // input layout
      "xck",                       // filter layout
      "none",                      // group layout
      false,                       // Winograd OK?
      base_name + "_branch2a",     // name
      "ungrouped",                 // autogroup (ie for grouped)
      "none",                      // deriv (ie for transposed)
      {});
  auto relu_2a = op::relu(conv_2a + B[0]);
  auto conv_2b = op::convolution(  //
      relu_2a,                     // input
      W[1],                        // weights
      {1, 1},                      // strides
      {1, 1},                      // dilations
      {1, 1},                      // data dilations
      {},                          // filter shape (ie for transposed)
      1,                           // # of groups
      "same_upper",                // autopadding
      {},                          // manual padding
      "nxc",                       // input layout
      "xck",                       // filter layout
      "none",                      // group layout
      false,                       // Winograd OK?
      base_name + "_branch2b",     // name
      "ungrouped",                 // autogroup (ie for grouped)
      "none",                      // deriv (ie for transposed)
      {});
  auto relu_2b = op::relu(conv_2b + B[1]);
  auto conv_2c = op::convolution(  //
      relu_2b,                     // input
      W[2],                        // weights
      {1, 1},                      // strides
      {1, 1},                      // dilations
      {1, 1},                      // data dilations
      {},                          // filter shape (ie for transposed)
      1,                           // # of groups
      "valid",                     // autopadding
      {},                          // manual padding
      "nxc",                       // input layout
      "xck",                       // filter layout
      "none",                      // group layout
      false,                       // Winograd OK?
      base_name + "_branch2c",     // name
      "ungrouped",                 // autogroup (ie for grouped)
      "none",                      // deriv (ie for transposed)
      {});
  if (use_shortcut_conv) {
    auto conv_1 = op::convolution(  //
        I,                          // input
        W[3],                       // weights
        strides,                    // strides
        {1, 1},                     // dilations
        {1, 1},                     // data dilations
        {},                         // filter shape (ie for transposed)
        1,                          // # of groups
        "valid",                    // autopadding
        {},                         // manual padding
        "nxc",                      // input layout
        "xck",                      // filter layout
        "none",                     // group layout
        false,                      // Winograd OK?
        base_name + "_branch1",     // name
        "ungrouped",                // autogroup (ie for grouped)
        "none",                     // deriv (ie for transposed)
        {});
    return op::relu(conv_2c + B[2] + conv_1 + B[3]);
  } else {
    return op::relu(conv_2c + B[2] + I);
  }
}

std::vector<edsl::Tensor> weight_placeholders() {
  return {
      // conv1
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}),

      // block2a
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 64, 64}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {3, 3, 64, 64}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 64, 256}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 64, 256}),
      // block2b
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 256, 64}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {3, 3, 64, 64}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 64, 256}),
      // block2c
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 256, 64}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {3, 3, 64, 64}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 64, 256}),

      // block3a
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 256, 128}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {3, 3, 128, 128}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 128, 512}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 256, 512}),
      // block3b
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 512, 128}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {3, 3, 128, 128}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 128, 512}),
      // block3c
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 512, 128}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {3, 3, 128, 128}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 128, 512}),
      // block3d
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 512, 128}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {3, 3, 128, 128}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 128, 512}),

      // block4a
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 512, 256}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {3, 3, 256, 256}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 256, 1024}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 512, 1024}),
      // block4b
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 1024, 256}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {3, 3, 256, 256}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 256, 1024}),
      // block4c
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 1024, 256}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {3, 3, 256, 256}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 256, 1024}),
      // block4d
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 1024, 256}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {3, 3, 256, 256}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 256, 1024}),
      // block4e
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 1024, 256}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {3, 3, 256, 256}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 256, 1024}),
      // block4f
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 1024, 256}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {3, 3, 256, 256}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 256, 1024}),

      // block5a
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 1024, 512}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {3, 3, 512, 512}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 512, 2048}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 1024, 2048}),
      // block5b
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 2048, 512}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {3, 3, 512, 512}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 512, 2048}),
      // block5c
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 2048, 512}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {3, 3, 512, 512}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1, 1, 512, 2048}),

      // dense
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {2048, 1000}),
  };
}

std::vector<edsl::Tensor> bias_placeholders() {
  return {
      // conv1
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {64}),

      // block2a
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {64}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {64}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {256}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {256}),
      // block2b
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {64}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {64}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {256}),
      // block2c
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {64}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {64}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {256}),

      // block3a
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {128}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {128}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {512}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {512}),
      // block3b
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {128}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {128}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {512}),
      // block3c
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {128}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {128}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {512}),
      // block3d
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {128}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {128}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {512}),

      // block4a
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {256}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {256}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1024}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1024}),
      // block4b
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {256}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {256}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1024}),
      // block4c
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {256}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {256}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1024}),
      // block4d
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {256}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {256}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1024}),
      // block4e
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {256}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {256}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1024}),
      // block4f
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {256}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {256}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1024}),

      // block5a
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {512}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {512}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {2048}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {2048}),
      // block5b
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {512}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {512}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {2048}),
      // block5c
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {512}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {512}),
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {2048}),

      // dense
      edsl::Placeholder(PLAIDML_DATA_FLOAT32, {1000}),
  };
}

}  // namespace

struct resnet50 : public benchmark::Fixture {
  void SetUp(const benchmark::State& state) {  //
    plaidml::op::init();
  }

  edsl::Program build(                     //
      int64_t batch_size,                  //
      const edsl::Tensor& I,               //
      const std::vector<edsl::Tensor>& W,  //
      const std::vector<edsl::Tensor>& B) {
    auto W_conv1 = W[0];
    auto B_conv1 = B[0];
    auto conv1 = op::convolution(  //
        I,                         // input
        W_conv1,                   // weights
        {2, 2},                    // strides
        {1, 1},                    // dilations
        {1, 1},                    // data dilations
        {},                        // filter shape (ie for transposed)
        1,                         // # of groups
        "none",                    // autopadding
        {3, 3},                    // manual padding
        "nxc",                     // input layout
        "xck",                     // filter layout
        "none",                    // group layout
        false,                     // Winograd OK?
        "conv1",                   // name
        "ungrouped",               // autogroup (ie for grouped)
        "none",                    // deriv (ie for transposed)
        {});
    auto relu1 = op::relu(conv1 + B_conv1);
    auto pool1 = op::pool(  //
        relu1,              // input
        "max",              // pool mode
        {3, 3},             // pool shape
        {2, 2},             // strides
        "none",             // autopadding
        {1, 1},             // manual padding
        "nxc");             // input layout

    // 2
    std::vector<edsl::Tensor> W_block2a = {W[1], W[2], W[3], W[4]};
    std::vector<edsl::Tensor> B_block2a = {B[1], B[2], B[3], B[4]};
    auto block_2a = block(pool1, W_block2a, B_block2a, {1, 1}, true, "res2a");

    std::vector<edsl::Tensor> W_block2b = {W[5], W[6], W[7]};
    std::vector<edsl::Tensor> B_block2b = {B[5], B[6], B[7]};
    auto block_2b = block(block_2a, W_block2b, B_block2b, {1, 1}, false, "res2b");

    std::vector<edsl::Tensor> W_block2c = {W[8], W[9], W[10]};
    std::vector<edsl::Tensor> B_block2c = {B[8], B[9], B[10]};
    auto block_2c = block(block_2b, W_block2c, B_block2c, {1, 1}, false, "res2c");

    // 3
    std::vector<edsl::Tensor> W_block3a = {W[11], W[12], W[13], W[14]};
    std::vector<edsl::Tensor> B_block3a = {B[11], B[12], B[13], B[14]};
    auto block_3a = block(block_2c, W_block3a, B_block3a, {2, 2}, true, "res3a");

    std::vector<edsl::Tensor> W_block3b = {W[15], W[16], W[17]};
    std::vector<edsl::Tensor> B_block3b = {B[15], B[16], B[17]};
    auto block_3b = block(block_3a, W_block3b, B_block3b, {1, 1}, false, "res3b");

    std::vector<edsl::Tensor> W_block3c = {W[18], W[19], W[20]};
    std::vector<edsl::Tensor> B_block3c = {B[18], B[19], B[20]};
    auto block_3c = block(block_3b, W_block3c, B_block3c, {1, 1}, false, "res3c");

    std::vector<edsl::Tensor> W_block3d = {W[21], W[22], W[23]};
    std::vector<edsl::Tensor> B_block3d = {B[21], B[22], B[23]};
    auto block_3d = block(block_3c, W_block3d, B_block3d, {1, 1}, false, "res3d");

    // 4
    std::vector<edsl::Tensor> W_block4a = {W[24], W[25], W[26], W[27]};
    std::vector<edsl::Tensor> B_block4a = {B[24], B[25], B[26], B[27]};
    auto block_4a = block(block_3d, W_block4a, B_block4a, {2, 2}, true, "res4a");

    std::vector<edsl::Tensor> W_block4b = {W[28], W[29], W[30]};
    std::vector<edsl::Tensor> B_block4b = {B[28], B[29], B[30]};
    auto block_4b = block(block_4a, W_block4b, B_block4b, {1, 1}, false, "res4b");

    std::vector<edsl::Tensor> W_block4c = {W[31], W[32], W[33]};
    std::vector<edsl::Tensor> B_block4c = {B[31], B[32], B[33]};
    auto block_4c = block(block_4b, W_block4c, B_block4c, {1, 1}, false, "res4c");

    std::vector<edsl::Tensor> W_block4d = {W[34], W[35], W[36]};
    std::vector<edsl::Tensor> B_block4d = {B[34], B[35], B[36]};
    auto block_4d = block(block_4c, W_block4d, B_block4d, {1, 1}, false, "res4d");

    std::vector<edsl::Tensor> W_block4e = {W[37], W[38], W[39]};
    std::vector<edsl::Tensor> B_block4e = {B[37], B[38], B[39]};
    auto block_4e = block(block_4d, W_block4e, B_block4e, {1, 1}, false, "res4e");

    std::vector<edsl::Tensor> W_block4f = {W[40], W[41], W[42]};
    std::vector<edsl::Tensor> B_block4f = {B[40], B[41], B[42]};
    auto block_4f = block(block_4e, W_block4f, B_block4f, {1, 1}, false, "res4f");

    // 5
    std::vector<edsl::Tensor> W_block5a = {W[43], W[44], W[45], W[46]};
    std::vector<edsl::Tensor> B_block5a = {B[43], B[44], B[45], B[46]};
    auto block_5a = block(block_4f, W_block5a, B_block5a, {2, 2}, true, "res5a");

    std::vector<edsl::Tensor> W_block5b = {W[47], W[48], W[49]};
    std::vector<edsl::Tensor> B_block5b = {B[47], B[48], B[49]};
    auto block_5b = block(block_5a, W_block5b, B_block5b, {1, 1}, false, "res5b");

    std::vector<edsl::Tensor> W_block5c = {W[50], W[51], W[52]};
    std::vector<edsl::Tensor> B_block5c = {B[50], B[51], B[52]};
    auto block_5c = block(block_5b, W_block5c, B_block5c, {1, 1}, false, "res5c");

    // End
    auto global_mean = op::mean(block_5c, edsl::make_tuple<int64_t>({1, 2}));
    auto W_dense = W[53];
    auto B_dense = B[53];
    auto dense = op::dot(global_mean, W_dense) + B_dense;
    auto softmax = op::softmax(dense, 1);
    return edsl::Program("resnet50", {softmax});
  }

  auto compile(int64_t batch_size = 1) {
    auto I = edsl::Placeholder(PLAIDML_DATA_FLOAT32, {batch_size, 224, 224, 3});
    auto W = weight_placeholders();
    auto B = bias_placeholders();
    auto program = build(batch_size, I, W, B);
    return exec::Binder(program).compile();
  }
};

BENCHMARK_DEFINE_F(resnet50, build)(benchmark::State& state) {  // NOLINT[runtime/references]
  compile()->run();
}

BENCHMARK_DEFINE_F(resnet50, compile)(benchmark::State& state) {  // NOLINT[runtime/references]
  for (auto _ : state) {
    compile();
  }
}

BENCHMARK_DEFINE_F(resnet50, run)(benchmark::State& state) {  // NOLINT[runtime/references]
  auto executable = compile();
  for (auto _ : state) {
    executable->run();
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_REGISTER_F(resnet50, build)->Unit(benchmark::kMillisecond)->Iterations(1);

BENCHMARK_REGISTER_F(resnet50, compile)->Unit(benchmark::kMillisecond);

// TODO: get HAL timer results, UseManualTime() instead of UseRealTime()
BENCHMARK_REGISTER_F(resnet50, run)->Unit(benchmark::kMillisecond)->UseRealTime();

}  // namespace networks::oplib
