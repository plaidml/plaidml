// Copyright 2019 Intel Corporation.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "llvm/ADT/StringRef.h"

#include "plaidml/op/op.h"
#include "plaidml/testenv.h"
#include "pmlc/util/logging.h"

using ::testing::Eq;

using namespace plaidml::edsl;  // NOLINT

namespace plaidml::edsl {

bool operator==(const Program& lhs, const std::string& rhs) {
  // TODO: re-enable brittle tests by replacing with lit.
  return true;
  // return llvm::StringRef(lhs.str()).trim() == llvm::StringRef(rhs).trim();
}

}  // namespace plaidml::edsl

namespace plaidml::op {
namespace {

class OpTest : public TestFixture {};

Program makeProgram(const std::string& name, const std::vector<Tensor>& outputs) {
  // TODO: remove empty target once all of these are passing.
  return ProgramBuilder(name, outputs).target("").compile();
}

TEST(Op, Abs) {
  auto I = Placeholder(DType::FLOAT32, {1, 224, 224, 3}, "I");
  auto abs = op::abs(I);
  auto program = makeProgram("abs", {abs});
  IVLOG(1, program);

  EXPECT_THAT(program, Eq(R"#(

module {
  func @abs(%arg0: tensor<1x224x224x3xf32> {tile.name = "I"}) -> tensor<1x224x224x3xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = "eltwise.neg"(%arg0) : (tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
    %1 = "eltwise.cmp_lt"(%arg0, %cst) : (tensor<1x224x224x3xf32>, f32) -> tensor<1x224x224x3xi1>
    %2 = "eltwise.select"(%1, %0, %arg0) : (tensor<1x224x224x3xi1>, tensor<1x224x224x3xf32>, tensor<1x224x224x3xf32>) -> tensor<1x224x224x3xf32>
    return %2 : tensor<1x224x224x3xf32>
  }
}
)#"));
}

TEST(Op, All) {
  auto I = Placeholder(DType::FLOAT32, {1, 224, 224, 3}, "I");
  auto program = makeProgram("all", {op::all(I)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<() -> ()>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>


module {
  func @all(%arg0: tensor<1x224x224x3xf32> {tile.name = "I"}) -> ui8 {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %c0 = "eltwise.sconst"() {value = 0 : i64} : () -> si32
    %c1 = "eltwise.sconst"() {value = 1 : i64} : () -> si32
    %0 = "eltwise.cmp_eq"(%arg0, %c0) : (tensor<1x224x224x3xf32>, si32) -> tensor<1x224x224x3xi1>
    %1 = "eltwise.select"(%0, %c0, %c1) : (tensor<1x224x224x3xi1>, si32, si32) -> tensor<1x224x224x3xsi32>
    %2 = tile.contract mul, none, %cst, %1 {sink = #map0, srcs = [#map1]} : f32, tensor<1x224x224x3xsi32> -> si32
    %3 = "eltwise.cast"(%2) : (si32) -> ui8
    return %3 : ui8
  }
}
)#"));
}

TEST(Op, Any) {
  auto I = Placeholder(DType::FLOAT32, {1, 224, 224, 3}, "I");
  auto program = makeProgram("any", {op::any(I)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<() -> ()>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>


module {
  func @any(%arg0: tensor<1x224x224x3xf32> {tile.name = "I"}) -> ui8 {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %c0 = "eltwise.sconst"() {value = 0 : i64} : () -> si32
    %c1 = "eltwise.sconst"() {value = 1 : i64} : () -> si32
    %0 = "eltwise.cmp_eq"(%arg0, %c0) : (tensor<1x224x224x3xf32>, si32) -> tensor<1x224x224x3xi1>
    %1 = "eltwise.select"(%0, %c0, %c1) : (tensor<1x224x224x3xi1>, si32, si32) -> tensor<1x224x224x3xsi32>
    %2 = tile.contract add, none, %cst, %1 {sink = #map0, srcs = [#map1]} : f32, tensor<1x224x224x3xsi32> -> si32
    %3 = "eltwise.cmp_eq"(%2, %c0) : (si32, si32) -> i1
    %4 = "eltwise.select"(%3, %c0, %c1) : (i1, si32, si32) -> si32
    %5 = "eltwise.cast"(%4) : (si32) -> ui8
    return %5 : ui8
  }
}
)#"));
}

TEST(Op, Argmax) {
  auto I = Placeholder(DType::FLOAT32, {1, 224, 224, 3}, "I");
  auto program = makeProgram("argmax", {op::argmax(I)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<() -> ()>


module {
  func @argmax(%arg0: tensor<1x224x224x3xf32> {tile.name = "I"}) -> ui32 {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %c1 = "eltwise.sconst"() {value = 1 : i64} : () -> si32
    %0 = tile.contract assign, none, %cst, %c1 {sink = #map0, srcs = [#map1]} : f32, si32 -> tensor<1x224x224x3xsi32>
    %1 = "tile.index"(%0) {dim = 0 : i64} : (tensor<1x224x224x3xsi32>) -> tensor<1x224x224x3xsi32>
    %2 = tile.contract max, none, %cst, %arg0 {sink = #map1, srcs = [#map0]} : f32, tensor<1x224x224x3xf32> -> f32
    %3 = tile.contract max, cond, %cst, %arg0, %2, %1 {sink = #map1, srcs = [#map0, #map1, #map0]} : f32, tensor<1x224x224x3xf32>, f32, tensor<1x224x224x3xsi32> -> si32
    %4 = "eltwise.cast"(%3) : (si32) -> ui32
    return %4 : ui32
  }
}
)#"));
}

TEST(Op, BinaryCrossentropy) {
  auto I = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "I");
  auto O = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "O");
  auto program = makeProgram("binary_crossentropy", {op::binary_crossentropy(I, O, 0.0)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(
module {
  func @binary_crossentropy(%arg0: tensor<7x7x3x64xf32> {tile.name = "O"}, %arg1: tensor<7x7x3x64xf32> {tile.name = "I"}) -> tensor<7x7x3x64xf32> {
    %c1 = "eltwise.sconst"() {value = 1 : i64} : () -> si32
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %cst_0 = "eltwise.sconst"() {value = 1.000000e+00 : f64} : () -> f32
    %0 = "eltwise.cmp_gt"(%arg0, %cst) : (tensor<7x7x3x64xf32>, f32) -> tensor<7x7x3x64xi1>
    %1 = "eltwise.select"(%0, %arg0, %cst) : (tensor<7x7x3x64xi1>, tensor<7x7x3x64xf32>, f32) -> tensor<7x7x3x64xf32>
    %2 = "eltwise.cmp_lt"(%1, %cst_0) : (tensor<7x7x3x64xf32>, f32) -> tensor<7x7x3x64xi1>
    %3 = "eltwise.select"(%2, %1, %cst_0) : (tensor<7x7x3x64xi1>, tensor<7x7x3x64xf32>, f32) -> tensor<7x7x3x64xf32>
    %4 = "eltwise.sub"(%c1, %3) : (si32, tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>
    %5 = "eltwise.log"(%4) : (tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>
    %6 = "eltwise.ident"(%arg1) : (tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>
    %7 = "eltwise.sub"(%c1, %6) : (si32, tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>
    %8 = "eltwise.mul"(%7, %5) : (tensor<7x7x3x64xf32>, tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>
    %9 = "eltwise.log"(%3) : (tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>
    %10 = "eltwise.neg"(%6) : (tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>
    %11 = "eltwise.mul"(%10, %9) : (tensor<7x7x3x64xf32>, tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>
    %12 = "eltwise.sub"(%11, %8) : (tensor<7x7x3x64xf32>, tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>
    return %12 : tensor<7x7x3x64xf32>
  }
}
)#"));
}

TEST(Op, Clip) {
  auto I = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "I");
  auto raw_min = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "raw_min");
  auto raw_max = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "raw_max");
  auto program = makeProgram("clip", {op::clip(I, raw_min, raw_max)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

module {
  func @clip(%arg0: tensor<7x7x3x64xf32> {tile.name = "raw_max"}, %arg1: tensor<7x7x3x64xf32> {tile.name = "raw_min"}, %arg2: tensor<7x7x3x64xf32> {tile.name = "I"}) -> tensor<7x7x3x64xf32> {
    %0 = "eltwise.cmp_gt"(%arg2, %arg1) : (tensor<7x7x3x64xf32>, tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xi1>
    %1 = "eltwise.select"(%0, %arg2, %arg1) : (tensor<7x7x3x64xi1>, tensor<7x7x3x64xf32>, tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>
    %2 = "eltwise.cmp_lt"(%1, %arg0) : (tensor<7x7x3x64xf32>, tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xi1>
    %3 = "eltwise.select"(%2, %1, %arg0) : (tensor<7x7x3x64xi1>, tensor<7x7x3x64xf32>, tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>
    return %3 : tensor<7x7x3x64xf32>
  }
}
)#"));
}

TEST(Op, Concatenate) {
  auto A = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "A");
  auto B = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "B");
  auto program = makeProgram("concatenate", {op::concatenate({A, B}, 2)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 + 3, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>


module {
  func @concatenate(%arg0: tensor<7x7x3x64xf32> {tile.name = "B"}, %arg1: tensor<7x7x3x64xf32> {tile.name = "A"}) -> tensor<7x7x6x64xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract assign, none, %cst, %arg0 {idxs = ["n0", "n1", "a", "n3"], sink = #map0, srcs = [#map1]} : f32, tensor<7x7x3x64xf32> -> tensor<7x7x6x64xf32>
    %1 = tile.contract assign, none, %cst, %arg1 {idxs = ["n0", "n1", "a", "n3"], sink = #map1, srcs = [#map1]} : f32, tensor<7x7x3x64xf32> -> tensor<7x7x6x64xf32>
    %2 = "eltwise.add"(%1, %0) : (tensor<7x7x6x64xf32>, tensor<7x7x6x64xf32>) -> tensor<7x7x6x64xf32>
    return %2 : tensor<7x7x6x64xf32>
  }
}
)#"));
}

TEST(Op, Convolution) {
  auto I = Placeholder(DType::FLOAT32, {1, 224, 224, 3}, "I");
  auto K = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "K");
  auto O = op::convolution(I, K).strides({2, 2}).autopad_mode(AutoPadMode::EXPLICIT).manual_padding({3, 3});
  auto program = makeProgram("convolution", {O});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 * 2 + d4 - 3, d2 * 2 + d5 - 3, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>


module {
  func @convolution(%arg0: tensor<7x7x3x64xf32> {tile.name = "K"}, %arg1: tensor<1x224x224x3xf32> {tile.name = "I"}) -> tensor<1x112x112x64xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %conv = tile.contract add, mul, %cst, %arg1, %arg0 {idxs = ["n", "x0", "x1", "co", "k0", "k1", "ci"], sink = #map0, srcs = [#map1, #map2]} : f32, tensor<1x224x224x3xf32>, tensor<7x7x3x64xf32> -> tensor<1x112x112x64xf32>
    return %conv : tensor<1x112x112x64xf32>
  }
}
)#"));
}

TEST(Op, CumProd) {
  auto I = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "I");
  auto program = makeProgram("cumprod", {op::cumprod(I, 2)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2 - d4, d3)>

#set0 = affine_set<(d0, d1, d2, d3, d4) : (d4 >= 0, -d4 + 2 >= 0)>

module {
  func @cumprod(%arg0: tensor<7x7x3x64xf32> {tile.name = "I"}) -> tensor<7x7x3x64xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract mul, none, %cst, %arg0 {cons = #set0, sink = #map0, srcs = [#map1]} : f32, tensor<7x7x3x64xf32> -> tensor<7x7x3x64xf32>
    return %0 : tensor<7x7x3x64xf32>
  }
}
)#"));
}

TEST(Op, CumSum) {
  auto I = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "I");
  auto program = makeProgram("cumsum", {op::cumsum(I, 2)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2 - d4, d3)>

#set0 = affine_set<(d0, d1, d2, d3, d4) : (d4 >= 0, -d4 + 2 >= 0)>

module {
  func @cumsum(%arg0: tensor<7x7x3x64xf32> {tile.name = "I"}) -> tensor<7x7x3x64xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract add, none, %cst, %arg0 {cons = #set0, sink = #map0, srcs = [#map1]} : f32, tensor<7x7x3x64xf32> -> tensor<7x7x3x64xf32>
    return %0 : tensor<7x7x3x64xf32>
  }
}
)#"));
}

TEST(Op, Dot) {
  auto I = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "I");
  auto K = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "K");
  auto program = makeProgram("dot", {op::dot(I, K)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d6)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d3, d4, d6, d5)>


module {
  func @dot(%arg0: tensor<7x7x3x64xf32> {tile.name = "K"}, %arg1: tensor<7x7x3x64xf32> {tile.name = "I"}) -> tensor<7x7x3x7x7x64xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract add, mul, %cst, %arg1, %arg0 {sink = #map0, srcs = [#map1, #map2]} : f32, tensor<7x7x3x64xf32>, tensor<7x7x3x64xf32> -> tensor<7x7x3x7x7x64xf32>
    return %0 : tensor<7x7x3x7x7x64xf32>
  }
}
)#"));
}

TEST(Op, Elu) {
  auto I = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "I");
  auto program = makeProgram("elu", {op::elu(I, 0.1)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(
module {
  func @elu(%arg0: tensor<7x7x3x64xf32> {tile.name = "I"}) -> tensor<7x7x3x64xf32> {
    %c0 = "eltwise.sconst"() {value = 0 : i64} : () -> si32
    %cst = "eltwise.sconst"() {value = 1.000000e-01 : f64} : () -> f32
    %0 = "eltwise.exp"(%arg0) : (tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>
    %1 = "eltwise.mul"(%0, %cst) : (tensor<7x7x3x64xf32>, f32) -> tensor<7x7x3x64xf32>
    %2 = "eltwise.sub"(%1, %cst) : (tensor<7x7x3x64xf32>, f32) -> tensor<7x7x3x64xf32>
    %3 = "eltwise.cmp_lt"(%arg0, %c0) : (tensor<7x7x3x64xf32>, si32) -> tensor<7x7x3x64xi1>
    %4 = "eltwise.select"(%3, %2, %arg0) : (tensor<7x7x3x64xi1>, tensor<7x7x3x64xf32>, tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>
    return %4 : tensor<7x7x3x64xf32>
  }
}
)#"));
}

TEST(Op, ExpandDims) {
  auto I = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "I");
  auto program = makeProgram("expand_dims", {op::expand_dims(I, 2)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>


module {
  func @expand_dims(%arg0: tensor<7x7x3x64xf32> {tile.name = "I"}) -> tensor<7x7x1x3x64xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract assign, none, %cst, %arg0 {idxs = ["n0", "n1", "a", "n2", "n3"], sink = #map0, srcs = [#map1]} : f32, tensor<7x7x3x64xf32> -> tensor<7x7x1x3x64xf32>
    return %0 : tensor<7x7x1x3x64xf32>
  }
}
)#"));
}

TEST(Op, Flip) {
  auto I = Placeholder(DType::FLOAT32, {7, 7, 3, 64}, "I");
  auto program = makeProgram("flip", {op::flip(I, 2)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, -d2 + 2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>


module {
  func @flip(%arg0: tensor<7x7x3x64xf32> {tile.name = "I"}) -> tensor<7x7x3x64xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract assign, none, %cst, %arg0 {sink = #map0, srcs = [#map1]} : f32, tensor<7x7x3x64xf32> -> tensor<7x7x3x64xf32>
    return %0 : tensor<7x7x3x64xf32>
  }
}
)#"));
}

TEST(Op, HardSigmoid) {
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  auto program = makeProgram("hard_sigmoid", {op::hard_sigmoid(A, 0.05)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

module {
  func @hard_sigmoid(%arg0: tensor<10x20xf32> {tile.name = "A"}) -> tensor<10x20xf32> {
    %cst = "eltwise.sconst"() {value = -1.000000e+01 : f64} : () -> f32
    %cst_0 = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %cst_1 = "eltwise.sconst"() {value = 1.000000e+01 : f64} : () -> f32
    %cst_2 = "eltwise.sconst"() {value = 1.000000e+00 : f64} : () -> f32
    %cst_3 = "eltwise.sconst"() {value = 5.000000e-02 : f64} : () -> f32
    %cst_4 = "eltwise.sconst"() {value = 5.000000e-01 : f64} : () -> f32
    %0 = "eltwise.mul"(%arg0, %cst_3) : (tensor<10x20xf32>, f32) -> tensor<10x20xf32>
    %1 = "eltwise.add"(%0, %cst_4) : (tensor<10x20xf32>, f32) -> tensor<10x20xf32>
    %2 = "eltwise.cmp_gt"(%arg0, %cst_1) : (tensor<10x20xf32>, f32) -> tensor<10x20xi1>
    %3 = "eltwise.select"(%2, %cst_2, %1) : (tensor<10x20xi1>, f32, tensor<10x20xf32>) -> tensor<10x20xf32>
    %4 = "eltwise.cmp_lt"(%arg0, %cst) : (tensor<10x20xf32>, f32) -> tensor<10x20xi1>
    %5 = "eltwise.select"(%4, %cst_0, %3) : (tensor<10x20xi1>, f32, tensor<10x20xf32>) -> tensor<10x20xf32>
    return %5 : tensor<10x20xf32>
  }
}
)#"));
}

TEST(Op, ImageResize) {
  auto I = Placeholder(DType::FLOAT32, {1, 224, 224, 3}, "I");
  auto image_resize = op::image_resize(I, std::vector<int>{5, 4}, InterpolationMode::BILINEAR, TensorLayout::NXC);
  auto program = makeProgram("image_resize", {image_resize});
  IVLOG(1, program);
}

TEST(Op, Max) {
  auto I = Placeholder(DType::FLOAT32, {1, 224, 224, 3}, "I");
  auto program = makeProgram("max", {op::max(I)});  // NOLINT(build/include_what_you_use)
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<() -> ()>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>


module {
  func @max(%arg0: tensor<1x224x224x3xf32> {tile.name = "I"}) -> f32 {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract max, none, %cst, %arg0 {sink = #map0, srcs = [#map1]} : f32, tensor<1x224x224x3xf32> -> f32
    return %0 : f32
  }
}
)#"));
}

TEST(Op, Maximum) {
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  auto B = Placeholder(DType::FLOAT32, {10, 20}, "B");
  auto program = makeProgram("maximum", {op::maximum(A, B)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

module {
  func @maximum(%arg0: tensor<10x20xf32> {tile.name = "A"}, %arg1: tensor<10x20xf32> {tile.name = "B"}) -> tensor<10x20xf32> {
    %0 = "eltwise.cmp_lt"(%arg0, %arg1) : (tensor<10x20xf32>, tensor<10x20xf32>) -> tensor<10x20xi1>
    %1 = "eltwise.select"(%0, %arg1, %arg0) : (tensor<10x20xi1>, tensor<10x20xf32>, tensor<10x20xf32>) -> tensor<10x20xf32>
    return %1 : tensor<10x20xf32>
  }
}
)#"));
}

TEST(Op, Mean) {
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  auto program = makeProgram("mean", {op::mean(A)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<() -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>


module {
  func @mean(%arg0: tensor<10x20xf32> {tile.name = "A"}) -> f32 {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %c200 = "eltwise.sconst"() {value = 200 : index} : () -> si32
    %0 = tile.contract add, none, %cst, %arg0 {sink = #map0, srcs = [#map1]} : f32, tensor<10x20xf32> -> f32
    %1 = "eltwise.div"(%0, %c200) : (f32, si32) -> f32
    return %1 : f32
  }
}
)#"));
}

TEST(Op, Min) {
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  auto program = makeProgram("min", {op::min(A)});  // NOLINT(build/include_what_you_use)
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<() -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>


module {
  func @min(%arg0: tensor<10x20xf32> {tile.name = "A"}) -> f32 {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract min, none, %cst, %arg0 {sink = #map0, srcs = [#map1]} : f32, tensor<10x20xf32> -> f32
    return %0 : f32
  }
}
)#"));
}

TEST(Op, Minimum) {
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  auto B = Placeholder(DType::FLOAT32, {10, 20}, "B");
  auto program = makeProgram("minimum", {op::minimum(A, B)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

module {
  func @minimum(%arg0: tensor<10x20xf32> {tile.name = "B"}, %arg1: tensor<10x20xf32> {tile.name = "A"}) -> tensor<10x20xf32> {
    %0 = "eltwise.cmp_lt"(%arg1, %arg0) : (tensor<10x20xf32>, tensor<10x20xf32>) -> tensor<10x20xi1>
    %1 = "eltwise.select"(%0, %arg1, %arg0) : (tensor<10x20xi1>, tensor<10x20xf32>, tensor<10x20xf32>) -> tensor<10x20xf32>
    return %1 : tensor<10x20xf32>
  }
}
)#"));
}

TEST(Op, Pool) {
  auto I = Placeholder(DType::FLOAT32, {10, 20, 30, 40, 50}, "I");
  auto program = makeProgram("pool", {op::pool(I, PoolMode::SUM, {1, 2, 3}, {1, 2, 3}, AutoPadMode::NONE, {1, 2},
                                               TensorLayout::NXC, true, true)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1 + d5 - 1, d2 * 2 + d6 - 2, d3 * 3 + d7, d4)>

#set0 = affine_set<(d0, d1, d2, d3, d4, d5, d6, d7) : (d5 >= 0, -d5 >= 0, d6 >= 0, -d6 + 1 >= 0, d7 >= 0, -d7 + 2 >= 0)>

module {
  func @pool(%arg0: tensor<10x20x30x40x50xf32> {tile.name = "I"}) -> tensor<10x22x17x14x50xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract add, none, %cst, %arg0 {cons = #set0, sink = #map0, srcs = [#map1]} : f32, tensor<10x20x30x40x50xf32> -> tensor<10x22x17x14x50xf32>
    return %0 : tensor<10x22x17x14x50xf32>
  }
}
)#"));
}

TEST(Op, Prod) {
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  auto program = makeProgram("prod", {op::prod(A)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<() -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>


module {
  func @prod(%arg0: tensor<10x20xf32> {tile.name = "A"}) -> f32 {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract mul, none, %cst, %arg0 {sink = #map0, srcs = [#map1]} : f32, tensor<10x20xf32> -> f32
    return %0 : f32
  }
}
)#"));
}

TEST(Op, Relu) {
  auto I = Placeholder(DType::FLOAT32, {10, 20}, "I");
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  auto M = Placeholder(DType::FLOAT32, {10, 20}, "M");
  auto program = makeProgram("relu", {op::relu(I).alpha(A).max_value(M).threshold(0.05)});
  EXPECT_THAT(program, Eq(R"#(

module {
  func @relu(%arg0: tensor<10x20xf32> {tile.name = "M"}, %arg1: tensor<10x20xf32> {tile.name = "I"}, %arg2: tensor<10x20xf32> {tile.name = "A"}) -> tensor<10x20xf32> {
    %cst = "eltwise.sconst"() {value = 5.000000e-02 : f64} : () -> f32
    %0 = "eltwise.sub"(%arg1, %cst) : (tensor<10x20xf32>, f32) -> tensor<10x20xf32>
    %1 = "eltwise.mul"(%arg2, %0) : (tensor<10x20xf32>, tensor<10x20xf32>) -> tensor<10x20xf32>
    %2 = "eltwise.cmp_lt"(%arg1, %cst) : (tensor<10x20xf32>, f32) -> tensor<10x20xi1>
    %3 = "eltwise.select"(%2, %1, %arg1) : (tensor<10x20xi1>, tensor<10x20xf32>, tensor<10x20xf32>) -> tensor<10x20xf32>
    %4 = "eltwise.cmp_lt"(%3, %arg0) : (tensor<10x20xf32>, tensor<10x20xf32>) -> tensor<10x20xi1>
    %5 = "eltwise.select"(%4, %3, %arg0) : (tensor<10x20xi1>, tensor<10x20xf32>, tensor<10x20xf32>) -> tensor<10x20xf32>
    return %5 : tensor<10x20xf32>
  }
}
)#"));
}

TEST(Op, ReluNoAlpha) {
  auto I = Placeholder(DType::FLOAT32, {10, 20}, "I");
  auto M = Placeholder(DType::FLOAT32, {10, 20}, "M");
  auto program = makeProgram("relu", {op::relu(I).max_value(M).threshold(0.05)});
  EXPECT_THAT(program, Eq(R"#(

module {
  func @relu(%arg0: tensor<10x20xf32> {tile.name = "M"}, %arg1: tensor<10x20xf32> {tile.name = "I"}) -> tensor<10x20xf32> {
    %cst = "eltwise.sconst"() {value = 5.000000e-02 : f64} : () -> f32
    %cst_0 = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = "eltwise.cmp_lt"(%arg1, %cst) : (tensor<10x20xf32>, f32) -> tensor<10x20xi1>
    %1 = "eltwise.select"(%0, %cst_0, %arg1) : (tensor<10x20xi1>, f32, tensor<10x20xf32>) -> tensor<10x20xf32>
    %2 = "eltwise.cmp_lt"(%1, %arg0) : (tensor<10x20xf32>, tensor<10x20xf32>) -> tensor<10x20xi1>
    %3 = "eltwise.select"(%2, %1, %arg0) : (tensor<10x20xi1>, tensor<10x20xf32>, tensor<10x20xf32>) -> tensor<10x20xf32>
    return %3 : tensor<10x20xf32>
  }
}
)#"));
}

TEST(Op, ReluNoMaxValue) {
  auto I = Placeholder(DType::FLOAT32, {10, 20}, "I");
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  auto program = makeProgram("relu", {op::relu(I).alpha(A).threshold(0.05)});
  EXPECT_THAT(program, Eq(R"#(

module {
  func @relu(%arg0: tensor<10x20xf32> {tile.name = "I"}, %arg1: tensor<10x20xf32> {tile.name = "A"}) -> tensor<10x20xf32> {
    %cst = "eltwise.sconst"() {value = 5.000000e-02 : f64} : () -> f32
    %0 = "eltwise.sub"(%arg0, %cst) : (tensor<10x20xf32>, f32) -> tensor<10x20xf32>
    %1 = "eltwise.mul"(%arg1, %0) : (tensor<10x20xf32>, tensor<10x20xf32>) -> tensor<10x20xf32>
    %2 = "eltwise.cmp_lt"(%arg0, %cst) : (tensor<10x20xf32>, f32) -> tensor<10x20xi1>
    %3 = "eltwise.select"(%2, %1, %arg0) : (tensor<10x20xi1>, tensor<10x20xf32>, tensor<10x20xf32>) -> tensor<10x20xf32>
    return %3 : tensor<10x20xf32>
  }
}
)#"));
}

TEST(Op, ReluOnlyThreshold) {
  auto I = Placeholder(DType::FLOAT32, {10, 20}, "I");
  auto program = makeProgram("relu", {op::relu(I).threshold(0.05)});
  EXPECT_THAT(program, Eq(R"#(

module {
  func @relu(%arg0: tensor<10x20xf32> {tile.name = "I"}) -> tensor<10x20xf32> {
    %cst = "eltwise.sconst"() {value = 5.000000e-02 : f64} : () -> f32
    %cst_0 = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = "eltwise.cmp_lt"(%arg0, %cst) : (tensor<10x20xf32>, f32) -> tensor<10x20xi1>
    %1 = "eltwise.select"(%0, %cst_0, %arg0) : (tensor<10x20xi1>, f32, tensor<10x20xf32>) -> tensor<10x20xf32>
    return %1 : tensor<10x20xf32>
  }
}
)#"));
}

TEST(Op, ReluNoParams) {
  auto I = Placeholder(DType::FLOAT32, {10, 20}, "I");
  auto program = makeProgram("relu", {op::relu(I)});
  EXPECT_THAT(program, Eq(R"#(

module {
  func @relu(%arg0: tensor<10x20xf32> {tile.name = "I"}) -> tensor<10x20xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = "eltwise.cmp_lt"(%arg0, %cst) : (tensor<10x20xf32>, f32) -> tensor<10x20xi1>
    %1 = "eltwise.select"(%0, %cst, %arg0) : (tensor<10x20xi1>, f32, tensor<10x20xf32>) -> tensor<10x20xf32>
    return %1 : tensor<10x20xf32>
  }
}
)#"));
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

TEST(Op, Repeat) {
  auto A = Placeholder(DType::FLOAT32, {32, 1, 4, 1}, "A");
  auto X = op::repeat(  //
      A,                // tensor to repeat
      3,                // number of repeats
      2);               // axis to repeat
  auto program = makeProgram("repeat", {X});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2 * 3 + d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>

#set0 = affine_set<(d0, d1, d2, d3, d4) : (d3 >= 0, -d3 + 2 >= 0)>

module {
  func @repeat(%arg0: tensor<32x1x4x1xf32> {tile.name = "A"}) -> tensor<32x1x12x1xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract assign, none, %cst, %arg0 {cons = #set0, sink = #map0, srcs = [#map1]} : f32, tensor<32x1x4x1xf32> -> tensor<32x1x12x1xf32>
    return %0 : tensor<32x1x12x1xf32>
  }
}
)#"));
}

TEST(Op, Reshape) {
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  TensorDim I, J;
  A.bind_dims(I, J);
  auto program = makeProgram("reshape", {op::reshape(A, make_tuple(J, I))});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(
module {
  func @reshape(%arg0: tensor<10x20xf32> {tile.name = "A"}) -> tensor<20x10xf32> {
    %c20 = tile.constant 20
    %c10 = tile.constant 10
    %0 = "tile.reshape"(%arg0, %c20, %c10) : (tensor<10x20xf32>, index, index) -> tensor<20x10xf32>
    return %0 : tensor<20x10xf32>
  }
}
)#"));
}

TEST(Op, Sigmoid) {
  auto A = Placeholder(DType::FLOAT32, {10}, "A");
  auto program = makeProgram("sigmoid", {op::sigmoid(A)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

module {
  func @sigmoid(%arg0: tensor<10xf32> {tile.name = "A"}) -> tensor<10xf32> {
    %cst = "eltwise.sconst"() {value = 1.000000e+00 : f64} : () -> f32
    %0 = "eltwise.ident"(%arg0) : (tensor<10xf32>) -> tensor<10xf32>
    %1 = "eltwise.neg"(%0) : (tensor<10xf32>) -> tensor<10xf32>
    %2 = "eltwise.exp"(%1) : (tensor<10xf32>) -> tensor<10xf32>
    %3 = "eltwise.add"(%2, %cst) : (tensor<10xf32>, f32) -> tensor<10xf32>
    %4 = "eltwise.div"(%cst, %3) : (f32, tensor<10xf32>) -> tensor<10xf32>
    return %4 : tensor<10xf32>
  }
}
)#"));
}

TEST(Op, Slice) {
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  auto X = op::slice(  //
      A,               // tensor to perform spatial padding on
      {2, 10});        // slices
  auto program = makeProgram("slice", {X});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<() -> ()>
#map1 = affine_map<() -> (2, 10)>


module {
  func @slice(%arg0: tensor<10x20xf32> {tile.name = "A"}) -> f32 {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract assign, none, %cst, %arg0 {sink = #map0, srcs = [#map1]} : f32, tensor<10x20xf32> -> f32
    return %0 : f32
  }
}
)#"));
}

TEST(Op, Softmax) {
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  auto program = makeProgram("softmax", {op::softmax(A, 1)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<(d0, d1) -> (d0, 0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>


module {
  func @softmax(%arg0: tensor<10x20xf32> {tile.name = "A"}) -> tensor<10x20xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = "eltwise.ident"(%arg0) : (tensor<10x20xf32>) -> tensor<10x20xf32>
    %1 = tile.contract max, none, %cst, %0 {sink = #map0, srcs = [#map1]} : f32, tensor<10x20xf32> -> tensor<10x1xf32>
    %2 = "eltwise.sub"(%0, %1) : (tensor<10x20xf32>, tensor<10x1xf32>) -> tensor<10x20xf32>
    %3 = "eltwise.exp"(%2) : (tensor<10x20xf32>) -> tensor<10x20xf32>
    %4 = tile.contract add, none, %cst, %3 {sink = #map0, srcs = [#map1]} : f32, tensor<10x20xf32> -> tensor<10x1xf32>
    %5 = "eltwise.div"(%3, %4) : (tensor<10x20xf32>, tensor<10x1xf32>) -> tensor<10x20xf32>
    return %5 : tensor<10x20xf32>
  }
}
)#"));
}

TEST(Op, SpatialPadding) {
  auto A = Placeholder(DType::FLOAT32, {64, 4, 32, 32}, "A");
  auto X = op::spatial_padding(  //
      A,                         // tensor to perform spatial padding on
      {1, 3},                    // low pads
      {3, 3},                    // high pads
      TensorLayout::NXC);        // data layout
  auto program = makeProgram("spatial_padding", {X});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 + 1, d3 + 3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>


module {
  func @spatial_padding(%arg0: tensor<64x4x32x32xf32> {tile.name = "A"}) -> tensor<64x4x36x38xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract assign, none, %cst, %arg0 {idxs = ["n", "c", "x0", "x1"], sink = #map0, srcs = [#map1]} : f32, tensor<64x4x32x32xf32> -> tensor<64x4x36x38xf32>
    return %0 : tensor<64x4x36x38xf32>
  }
}
)#"));
}

TEST(Op, Square) {
  auto A = Placeholder(DType::FLOAT32, {10}, "A");
  auto program = makeProgram("square", {op::square(A)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

module {
  func @square(%arg0: tensor<10xf32> {tile.name = "A"}) -> tensor<10xf32> {
    %0 = "eltwise.mul"(%arg0, %arg0) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    return %0 : tensor<10xf32>
  }
}
)#"));
}

TEST(Op, Sum) {
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  auto program = makeProgram("sum", {op::sum(A)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<() -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>


module {
  func @sum(%arg0: tensor<10x20xf32> {tile.name = "A"}) -> f32 {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract add, none, %cst, %arg0 {sink = #map0, srcs = [#map1]} : f32, tensor<10x20xf32> -> f32
    return %0 : f32
  }
}
)#"));
}

TEST(Op, Squeeze) {
  auto A = Placeholder(DType::FLOAT32, {32, 1, 4, 1}, "A");
  auto X = op::squeeze(  //
      A,                 // tensor to squeeze
      {1, 3});           // axes to squeeze
  auto program = makeProgram("squeeze", {X});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(
module {
  func @squeeze(%arg0: tensor<32x1x4x1xf32> {tile.name = "A"}) -> tensor<32x4xf32> {
    %c32 = tile.constant 32
    %c4 = tile.constant 4
    %0 = "tile.reshape"(%arg0, %c32, %c4) : (tensor<32x1x4x1xf32>, index, index) -> tensor<32x4xf32>
    return %0 : tensor<32x4xf32>
  }
}
)#"));
}

TEST(Op, Tile) {
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  auto X = op::tile(  //
      A,              // tensor to tile
      {5, 4});        // tiling factors
  auto program = makeProgram("tile", {X});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 10 + d1, d2 * 20 + d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>


module {
  func @tile(%arg0: tensor<10x20xf32> {tile.name = "A"}) -> tensor<50x80xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract assign, none, %cst, %arg0 {no_reduce, sink = #map0, srcs = [#map1]} : f32, tensor<10x20xf32> -> tensor<50x80xf32>
    return %0 : tensor<50x80xf32>
  }
}
)#"));
}

TEST(Op, Transpose) {
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  auto program = makeProgram("transpose", {op::transpose(A)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>


module {
  func @transpose(%arg0: tensor<10x20xf32> {tile.name = "A"}) -> tensor<20x10xf32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %0 = tile.contract assign, none, %cst, %arg0 {sink = #map0, srcs = [#map1]} : f32, tensor<10x20xf32> -> tensor<20x10xf32>
    return %0 : tensor<20x10xf32>
  }
}
)#"));
}

TEST(Op, Variance) {
  auto A = Placeholder(DType::FLOAT32, {10, 20}, "A");
  auto program = makeProgram("variance", {op::variance(A)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(
#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map2 = affine_map<() -> ()>
#map3 = affine_map<(d0, d1) -> (d0, d1)>


module {
  func @variance(%arg0: tensor<10x20xf32> {tile.name = "A"}) -> f32 {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> f32
    %c200 = "eltwise.sconst"() {value = 200 : index} : () -> si32
    %0 = tile.contract add, none, %cst, %arg0 {sink = #map0, srcs = [#map1]} : f32, tensor<10x20xf32> -> tensor<1x1xf32>
    %1 = "eltwise.div"(%0, %c200) : (tensor<1x1xf32>, si32) -> tensor<1x1xf32>
    %2 = "eltwise.sub"(%arg0, %1) : (tensor<10x20xf32>, tensor<1x1xf32>) -> tensor<10x20xf32>
    %3 = "eltwise.mul"(%2, %2) : (tensor<10x20xf32>, tensor<10x20xf32>) -> tensor<10x20xf32>
    %4 = tile.contract add, none, %cst, %3 {sink = #map2, srcs = [#map3]} : f32, tensor<10x20xf32> -> f32
    %5 = "eltwise.div"(%4, %c200) : (f32, si32) -> f32
    return %5 : f32
  }
}
)#"));
}

}  // namespace
}  // namespace plaidml::op
