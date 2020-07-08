// Copyright 2019 Intel Corporation.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "llvm/ADT/StringRef.h"

#include "plaidml/op/op.h"
#include "plaidml/testenv.h"
#include "pmlc/util/logging.h"

using ::testing::Eq;

using namespace plaidml::edsl;  // NOLINT

namespace plaidml::op {
namespace {

class OpTest : public TestFixture {};

TEST_F(OpTest, Abs) {
  auto I = Placeholder(DType::FLOAT32, {1, 224, 224, 3}, "I");
  auto abs = op::abs(I);
  auto program = makeProgram("abs", {abs});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, All) {
  auto I = Placeholder(DType::FLOAT32, {1, 224, 224, 3}, "I");
  auto program = makeProgram("all", {op::all(I)});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, Any) {
  auto I = Placeholder(DType::FLOAT32, {1, 224, 224, 3}, "I");
  auto program = makeProgram("any", {op::any(I)});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, Argmax) {
  auto I = Placeholder(DType::FLOAT32, {1, 224, 224, 3}, "I");
  auto program = makeProgram("argmax", {op::argmax(I)});
  IVLOG(1, "\n" << program);
  runProgram(program);
}

TEST_F(OpTest, BinaryCrossentropy) {
  auto I = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "I");
  auto O = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "O");
  auto program = makeProgram("binary_crossentropy", {op::binary_crossentropy(I, O, 0.0)});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, BroadcastNoOp) {
  auto A = Placeholder(DType::FLOAT32, {3});
  std::vector<int> rshape = {3};
  std::vector<int> bdims = {0};
  auto C = broadcast(A, rshape, bdims);
  auto program = makeProgram("broadcast_nop", {C});

  std::vector<float> A_input = {0, 1, 2};
  checkProgram(program, {{A, A_input}}, {{C, A_input}});
}

TEST_F(OpTest, BroadcastScalar) {
  auto A = Placeholder(DType::FLOAT32, {});
  std::vector<int> rshape = {3, 4};
  std::vector<int> bdims = {};
  auto C = broadcast(A, rshape, bdims);
  auto program = makeProgram("broadcast_scalar", {C});

  std::vector<float> A_input = {3};
  std::vector<float> C_output = {3, 3, 3, 3,  //
                                 3, 3, 3, 3,  //
                                 3, 3, 3, 3};
  checkProgram(program, {{A, A_input}}, {{C, C_output}});
}

TEST_F(OpTest, BroadcastNoOpLarge) {
  auto A = Placeholder(DType::FLOAT32, {3, 4});
  std::vector<int> rshape = {3, 4};
  std::vector<int> bdims = {0, 1};
  auto C = broadcast(A, rshape, bdims);
  auto program = makeProgram("broadcast_nop_large", {C});

  std::vector<float> A_input = {0, 1, 2, 3,  //
                                0, 1, 2, 3,  //
                                0, 1, 2, 3};
  checkProgram(program, {{A, A_input}}, {{C, A_input}});
}

TEST_F(OpTest, BroadcastNumpy) {
  auto A = Placeholder(DType::FLOAT32, {1, 3, 3});
  std::vector<int> rshape = {1, 4, 3, 3};
  std::vector<int> bdims = {0, 2, 3};
  auto C = broadcast(A, rshape, bdims);
  auto program = makeProgram("broadcast_numpy", {C});

  std::vector<float> A_input = {0, 1, 2,  //
                                0, 1, 2,  //
                                0, 1, 2};
  std::vector<float> C_output = {0, 1, 2,  //
                                 0, 1, 2,  //
                                 0, 1, 2,  //
                                 0, 1, 2,  //
                                 0, 1, 2,  //
                                 0, 1, 2,  //
                                 0, 1, 2,  //
                                 0, 1, 2,  //
                                 0, 1, 2,  //
                                 0, 1, 2,  //
                                 0, 1, 2,  //
                                 0, 1, 2};
  checkProgram(program, {{A, A_input}}, {{C, C_output}});
}

TEST_F(OpTest, BroadcastNumpy2) {
  auto A = Placeholder(DType::FLOAT32, {3, 1});
  std::vector<int> rshape = {3, 4};
  std::vector<int> bdims = {0, 1};
  auto C = broadcast(A, rshape, bdims);
  auto program = makeProgram("broadcast_numpy_2", {C});

  std::vector<float> A_input = {0, 1, 2};
  std::vector<float> C_output = {0, 0, 0, 0,  //
                                 1, 1, 1, 1,  //
                                 2, 2, 2, 2};
  checkProgram(program, {{A, A_input}}, {{C, C_output}});
}

TEST_F(OpTest, BroadcastNumpy3) {
  auto A = Placeholder(DType::FLOAT32, {3});
  std::vector<int> rshape = {4, 3};
  std::vector<int> bdims = {1};
  auto C = broadcast(A, rshape, bdims);
  auto program = makeProgram("broadcast_numpy_3", {C});

  std::vector<float> A_input = {0, 1, 2};
  std::vector<float> C_output = {0, 1, 2,  //
                                 0, 1, 2,  //
                                 0, 1, 2,  //
                                 0, 1, 2};
  checkProgram(program, {{A, A_input}}, {{C, C_output}});
}

TEST_F(OpTest, BroadcastNonNumpy) {
  auto A = Placeholder(DType::FLOAT32, {3});
  std::vector<int> rshape = {3, 4};
  std::vector<int> bdims = {0};
  auto C = broadcast(A, rshape, bdims);
  auto program = makeProgram("broadcast_non_numpy", {C});

  std::vector<float> A_input = {0, 1, 2};
  std::vector<float> C_output = {0, 0, 0, 0,  //
                                 1, 1, 1, 1,  //
                                 2, 2, 2, 2};
  checkProgram(program, {{A, A_input}}, {{C, C_output}});
}

TEST_F(OpTest, Broadcast) {
  auto I = Placeholder(DType::FLOAT32, {1, 224, 224}, "I");
  std::vector<int> result_shape = {1, 224, 224, 3};
  std::vector<int> bcast_axes = {0, 1, 2};
  auto program = makeProgram("broadcast", {op::broadcast(I, result_shape, bcast_axes)});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, Clip) {
  auto I = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "I");
  auto raw_min = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "raw_min");
  auto raw_max = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "raw_max");
  auto program = makeProgram("clip", {op::clip(I, raw_min, raw_max)});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, Concatenate) {
  auto A = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "A");
  auto B = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "B");
  auto program = makeProgram("concatenate", {op::concatenate({A, B}, 2)});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, Convolution) {
  auto I = Placeholder(DType::FLOAT32, {1, 224, 224, 3}, "I");
  auto K = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "K");
  auto O = op::convolution(I, K).strides({2, 2}).autopad_mode(AutoPadMode::EXPLICIT).manual_padding({2, 2});
  auto program = makeProgram("convolution", {O});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, CumProd) {
  auto I = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "I");
  auto program = makeProgram("cumprod", {op::cumprod(I, 2)});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, CumSum) {
  auto I = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "I");
  auto program = makeProgram("cumsum", {op::cumsum(I, 2)});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, Dot) {
  auto I = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "I");
  auto K = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "K");
  auto program = makeProgram("dot", {op::dot(I, K)});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, Elu) {
  auto I = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "I");
  auto program = makeProgram("elu", {op::elu(I, 0.1)});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, Flip) {
  auto I = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "I");
  auto program = makeProgram("flip", {op::flip(I, 2)});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, HardSigmoid) {
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  auto program = makeProgram("hard_sigmoid", {op::hard_sigmoid(A, 0.05)});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, ImageResize) {
  auto I = Placeholder(DType::FLOAT32, {1, 224, 224, 3}, "I");
  auto image_resize = op::image_resize(I, std::vector<int>{5, 4}, InterpolationMode::BILINEAR, TensorLayout::NXC);
  auto program = makeProgram("image_resize", {image_resize});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, Max) {
  auto I = Placeholder(DType::FLOAT32, {1, 224, 224, 3}, "I");
  auto program = makeProgram("max", {op::max(I)});  // NOLINT(build/include_what_you_use)
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, Maximum) {
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  auto B = Placeholder(DType::FLOAT32, {10, 20}, "B");
  auto program = makeProgram("maximum", {op::maximum(A, B)});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, Mean) {
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  auto program = makeProgram("mean", {op::mean(A)});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, Min) {
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  auto program = makeProgram("min", {op::min(A)});  // NOLINT(build/include_what_you_use)
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, Minimum) {
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  auto B = Placeholder(DType::FLOAT32, {10, 20}, "B");
  auto program = makeProgram("minimum", {op::minimum(A, B)});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, L2Norm) {
  auto I = Placeholder(DType::FLOAT32, {10, 20}, "I");
  auto program = makeProgram("l2norm", {op::l2norm(I, {1}).epsilon(0.01).eps_mode(EpsMode::ADD)});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, Pool) {
  auto I = Placeholder(DType::FLOAT32, {10, 20, 30, 40, 50}, "I");
  auto program = makeProgram("pool", {op::pool(I, PoolMode::SUM, {1, 2, 3}, {1, 2, 3}, AutoPadMode::NONE, {1, 2},
                                               TensorLayout::NXC, true, true)});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, Prod) {
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  auto program = makeProgram("prod", {op::prod(A)});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, Relu) {
  auto I = Placeholder(DType::FLOAT32, {10, 20}, "I");
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  auto M = Placeholder(DType::FLOAT32, {10, 20}, "M");
  auto program = makeProgram("relu", {op::relu(I).alpha(A).max_value(M).threshold(0.05)});
  runProgram(program);
}

TEST_F(OpTest, ReluNoAlpha) {
  auto I = Placeholder(DType::FLOAT32, {10, 20}, "I");
  auto M = Placeholder(DType::FLOAT32, {10, 20}, "M");
  auto program = makeProgram("relu", {op::relu(I).max_value(M).threshold(0.05)});
  runProgram(program);
}

TEST_F(OpTest, ReluNoMaxValue) {
  auto I = Placeholder(DType::FLOAT32, {10, 20}, "I");
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  auto program = makeProgram("relu", {op::relu(I).alpha(A).threshold(0.05)});
  runProgram(program);
}

TEST_F(OpTest, ReluOnlyThreshold) {
  auto I = Placeholder(DType::FLOAT32, {10, 20}, "I");
  auto program = makeProgram("relu", {op::relu(I).threshold(0.05)});
  runProgram(program);
}

TEST_F(OpTest, ReluNoParams) {
  auto I = Placeholder(DType::FLOAT32, {10, 20}, "I");
  auto program = makeProgram("relu", {op::relu(I)});
  runProgram(program);
}

// See: https://leimao.github.io/blog/Reorg-Layer-Explained/
static std::vector<int64_t> reorgYoloRefImpl(const std::vector<int64_t>& I, unsigned N, unsigned C, unsigned H,
                                             unsigned W, unsigned stride, bool decrease) {
  std::vector<int64_t> O(I.size());
  auto C_out = C / (stride * stride);
  for (unsigned b = 0; b < N; b++) {
    for (unsigned k = 0; k < C; k++) {
      for (unsigned j = 0; j < H; j++) {
        for (unsigned i = 0; i < W; i++) {
          auto in_index = i + W * (j + H * (k + C * b));
          auto c2 = k % C_out;
          auto offset = k / C_out;
          auto w2 = i * stride + offset % stride;
          auto h2 = j * stride + offset / stride;
          auto out_index = w2 + W * stride * (h2 + H * stride * (c2 + C_out * b));
          if (decrease) {
            O[out_index] = I[in_index];
          } else {
            O[in_index] = I[out_index];
          }
        }
      }
    }
  }
  return O;
}

TEST_F(OpTest, ReorgYoloDecrease) {
  const unsigned N = 1, C = 4, H = 6, W = 6, S = 2;
  const bool decrease = true;

  auto I = Placeholder(DType::INT64, {N, C, H, W});
  auto O = op::reorg_yolo(I, S, decrease);
  auto program = ProgramBuilder("reorg_yolo", {O}).compile();
  IVLOG(1, "program:\n" << program);

  std::vector<int64_t> I_input(N * C * H * W);
  for (unsigned i = 0; i < I_input.size(); i++) {
    I_input[i] = i;
  }
  auto O_expected = reorgYoloRefImpl(I_input, N, C, H, W, S, decrease);
  IVLOG(1, "expected:\n" << O_expected);
  checkProgram(program, {{I, I_input}}, {{O, O_expected}});
}

TEST_F(OpTest, ReorgYoloIncrease) {
  const unsigned N = 1, C = 4, H = 6, W = 6, S = 2;
  const unsigned C_out = C * (S * S), H_out = H / S, W_out = W / S;
  const bool decrease = false;

  auto I = Placeholder(DType::INT64, {N, C, H, W});
  auto O = op::reorg_yolo(I, S, decrease);
  auto program = ProgramBuilder("reorg_yolo", {O}).compile();
  IVLOG(1, "program:\n" << program);

  std::vector<int64_t> I_input(N * C * H * W);
  for (unsigned i = 0; i < I_input.size(); i++) {
    I_input[i] = i;
  }
  auto O_expected = reorgYoloRefImpl(I_input, N, C_out, H_out, W_out, S, decrease);
  IVLOG(1, "expected:\n" << O_expected);
  checkProgram(program, {{I, I_input}}, {{O, O_expected}});
}

TEST_F(OpTest, Repeat) {
  auto A = Placeholder(DType::FLOAT32, {32, 1, 4, 1}, "A");
  auto X = op::repeat(  //
      A,                // tensor to repeat
      3,                // number of repeats
      2);               // axis to repeat
  auto program = makeProgram("repeat", {X});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, Reshape) {
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  TensorDim I, J;
  A.bind_dims(I, J);
  auto program = makeProgram("reshape", {op::reshape(A, make_tuple(J, I))});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, Sigmoid) {
  auto A = Placeholder(DType::FLOAT32, {10}, "A");
  auto program = makeProgram("sigmoid", {op::sigmoid(A)});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, Slice) {
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  auto X = op::slice(A)  //
               .add_dims({2, 10});
  auto program = makeProgram("slice", {X});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, Slice2) {
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  auto X = op::slice(A)  //
               .add_dim(/*start=*/0, /*stop=*/2)
               .add_dim(/*start=*/2, /*stop=*/8, /*step=*/2);
  auto program = makeProgram("slice", {X});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, Softmax) {
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  auto program = makeProgram("softmax", {op::softmax(A, 1)});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, SpatialPadding) {
  auto A = Placeholder(DType::FLOAT32, {64, 4, 32, 32}, "A");
  auto X = op::spatial_padding(  //
      A,                         // tensor to perform spatial padding on
      {1, 3},                    // low pads
      {3, 3},                    // high pads
      TensorLayout::NXC);        // data layout
  auto program = makeProgram("spatial_padding", {X});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, Square) {
  auto A = Placeholder(DType::FLOAT32, {10}, "A");
  auto program = makeProgram("square", {op::square(A)});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, Sum) {
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  auto program = makeProgram("sum", {op::sum(A)});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, Squeeze) {
  auto A = Placeholder(DType::UINT64, {3, 1, 4, 1}, "A");
  auto C = op::squeeze(  //
      A,                 // tensor to squeeze
      {1, 3});           // axes to squeeze
  auto program = makeProgram("squeeze", {C});
  IVLOG(1, program);

  std::vector<uint64_t> A_input = {0, 1, 2,  3,  //
                                   4, 5, 6,  7,  //
                                   8, 9, 10, 11};
  checkProgram(program, {{A, A_input}}, {{C, A_input}});
}

TEST_F(OpTest, Tile) {
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  auto X = op::tile(  //
      A,              // tensor to tile
      {5, 4});        // tiling factors
  auto program = makeProgram("tile", {X});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, Transpose) {
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  auto program = makeProgram("transpose", {op::transpose(A)});
  IVLOG(1, program);
  runProgram(program);
}

TEST_F(OpTest, Unsqueeze) {
  auto A = Placeholder(DType::UINT64, {3, 4}, "A");
  auto C = op::unsqueeze(  //
      A,                   // tensor to squeeze
      {0});                // axes to squeeze
  auto program = makeProgram("unsqueeze", {C});
  IVLOG(1, program);

  std::vector<uint64_t> A_input = {0, 1, 2,  3,  //
                                   4, 5, 6,  7,  //
                                   8, 9, 10, 11};
  checkProgram(program, {{A, A_input}}, {{C, A_input}});
}

TEST_F(OpTest, Variance) {
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  auto program = makeProgram("variance", {op::variance(A)});
  IVLOG(1, program);
  runProgram(program);
}

}  // namespace
}  // namespace plaidml::op
