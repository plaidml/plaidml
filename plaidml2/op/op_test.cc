// Copyright 2019 Intel Corporation.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "base/util/logging.h"
#include "plaidml2/op/op.h"

using ::testing::Eq;

using namespace plaidml::edsl;  // NOLINT

namespace plaidml::edsl {

bool operator==(const Program& lhs, const std::string& rhs) {  //
  return lhs.str() == rhs;
}

}  // namespace plaidml::edsl

namespace plaidml::op {
namespace {

TEST(Op, Abs) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {1, 224, 224, 3}, "I");
  auto abs = op::abs(I);
  Program program("abs", {abs});
  IVLOG(1, program);

  EXPECT_THAT(program, Eq(R"#(

!float = type !eltwise.float
!fp32 = type !eltwise.fp32
!bool = type !eltwise.bool
module {
  func @abs(%arg0: tensor<1x224x224x3x!eltwise.fp32> {tile.name = "I"}) -> tensor<1x224x224x3x!eltwise.fp32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !float
    %0 = "eltwise.neg"(%arg0) {type = !fp32} : (tensor<1x224x224x3x!eltwise.fp32>) -> tensor<1x224x224x3x!eltwise.fp32>
    %1 = "eltwise.cmp_lt"(%arg0, %cst) {type = !fp32} : (tensor<1x224x224x3x!eltwise.fp32>, !float) -> tensor<1x224x224x3x!eltwise.bool>
    %2 = "eltwise.select"(%1, %0, %arg0) {type = !fp32} : (tensor<1x224x224x3x!eltwise.bool>, tensor<1x224x224x3x!eltwise.fp32>, tensor<1x224x224x3x!eltwise.fp32>) -> tensor<1x224x224x3x!eltwise.fp32>
    return %2 : tensor<1x224x224x3x!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, All) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {1, 224, 224, 3}, "I");
  Program program("all", {op::all(I)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!int = type !eltwise.int
!fp32 = type !eltwise.fp32
!bool = type !eltwise.bool
!u8 = type !eltwise.u8
module {
  func @all(%arg0: tensor<1x224x224x3x!eltwise.fp32> {tile.name = "I"}) -> tensor<!eltwise.u8> {
    %c1 = "eltwise.sconst"() {value = 1 : i64} : () -> !int
    %c0 = "eltwise.sconst"() {value = 0 : i64} : () -> !int
    %0 = "eltwise.cmp_eq"(%arg0, %c0) {type = !fp32} : (tensor<1x224x224x3x!eltwise.fp32>, !int) -> tensor<1x224x224x3x!eltwise.bool>
    %1 = "eltwise.select"(%0, %c0, %c1) {type = !fp32} : (tensor<1x224x224x3x!eltwise.bool>, !int, !int) -> tensor<1x224x224x3x!eltwise.bool>
    %2 = "tile.domain"() ( {
    ^bb0(%arg1: !int, %arg2: !int, %arg3: !int, %arg4: !int):	// no predecessors
      %4 = "tile.src_idx_map"(%1, %arg4, %arg3, %arg2, %arg1) : (tensor<1x224x224x3x!eltwise.bool>, !int, !int, !int, !int) -> !tile.imap
      %5 = "tile.sink_idx_map"() : () -> !tile.imap
      %6 = "tile.size_map"() : () -> !tile.smap
      "tile.*(x)"(%6, %4, %5) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3"]} : () -> tensor<!eltwise.bool>
    %3 = "eltwise.cast"(%2) : (tensor<!eltwise.bool>) -> tensor<!eltwise.u8>
    return %3 : tensor<!eltwise.u8>
  }
}
)#"));
}

TEST(Op, Any) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {1, 224, 224, 3}, "I");
  Program program("any", {op::any(I)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!int = type !eltwise.int
!fp32 = type !eltwise.fp32
!bool = type !eltwise.bool
!u8 = type !eltwise.u8
module {
  func @any(%arg0: tensor<1x224x224x3x!eltwise.fp32> {tile.name = "I"}) -> tensor<!eltwise.u8> {
    %c1 = "eltwise.sconst"() {value = 1 : i64} : () -> !int
    %c0 = "eltwise.sconst"() {value = 0 : i64} : () -> !int
    %0 = "eltwise.cmp_eq"(%arg0, %c0) {type = !fp32} : (tensor<1x224x224x3x!eltwise.fp32>, !int) -> tensor<1x224x224x3x!eltwise.bool>
    %1 = "eltwise.select"(%0, %c0, %c1) {type = !fp32} : (tensor<1x224x224x3x!eltwise.bool>, !int, !int) -> tensor<1x224x224x3x!eltwise.bool>
    %2 = "tile.domain"() ( {
    ^bb0(%arg1: !int, %arg2: !int, %arg3: !int, %arg4: !int):	// no predecessors
      %6 = "tile.src_idx_map"(%1, %arg4, %arg3, %arg2, %arg1) : (tensor<1x224x224x3x!eltwise.bool>, !int, !int, !int, !int) -> !tile.imap
      %7 = "tile.sink_idx_map"() : () -> !tile.imap
      %8 = "tile.size_map"() : () -> !tile.smap
      "tile.+(x)"(%8, %6, %7) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3"]} : () -> tensor<!eltwise.bool>
    %3 = "eltwise.cmp_eq"(%2, %c0) {type = !fp32} : (tensor<!eltwise.bool>, !int) -> tensor<!eltwise.bool>
    %4 = "eltwise.select"(%3, %c0, %c1) {type = !fp32} : (tensor<!eltwise.bool>, !int, !int) -> tensor<!eltwise.bool>
    %5 = "eltwise.cast"(%4) : (tensor<!eltwise.bool>) -> tensor<!eltwise.u8>
    return %5 : tensor<!eltwise.u8>
  }
}
)#"));
}

TEST(Op, Argmax) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {1, 224, 224, 3}, "I");
  Program program("argmax", {op::argmax(I)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!int = type !eltwise.int
!fp32 = type !eltwise.fp32
!u32 = type !eltwise.u32
module {
  func @argmax(%arg0: tensor<1x224x224x3x!eltwise.fp32> {tile.name = "I"}) -> tensor<!eltwise.u32> {
    %c1 = "eltwise.sconst"() {value = 1 : i64} : () -> !int
    %c3 = "tile.affine_const"() {value = 3 : i64} : () -> !int
    %c224 = "tile.affine_const"() {value = 224 : i64} : () -> !int
    %c1_0 = "tile.affine_const"() {value = 1 : i64} : () -> !int
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !int, %arg2: !int, %arg3: !int, %arg4: !int):	// no predecessors
      %5 = "tile.src_idx_map"(%arg0, %arg4, %arg3, %arg2, %arg1) : (tensor<1x224x224x3x!eltwise.fp32>, !int, !int, !int, !int) -> !tile.imap
      %6 = "tile.sink_idx_map"() : () -> !tile.imap
      %7 = "tile.size_map"() : () -> !tile.smap
      "tile.>(x)"(%7, %5, %6) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3"]} : () -> tensor<!eltwise.fp32>
    %1 = "tile.domain"() ( {
    ^bb0(%arg1: !int, %arg2: !int, %arg3: !int, %arg4: !int):	// no predecessors
      %5 = "tile.src_idx_map"(%c1) : (!int) -> !tile.imap
      %6 = "tile.sink_idx_map"(%arg4, %arg3, %arg2, %arg1) : (!int, !int, !int, !int) -> !tile.imap
      %7 = "tile.size_map"(%c1_0, %c224, %c224, %c3) : (!int, !int, !int, !int) -> !tile.smap
      "tile.=(x)"(%7, %5, %6) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3"]} : () -> tensor<1x224x224x3x!eltwise.int>
    %2 = "tile.index"(%1) {dim = 0 : i64} : (tensor<1x224x224x3x!eltwise.int>) -> tensor<1x224x224x3x!eltwise.int>
    %3 = "tile.domain"() ( {
    ^bb0(%arg1: !int, %arg2: !int, %arg3: !int, %arg4: !int):	// no predecessors
      %5 = "tile.src_idx_map"(%arg0, %arg4, %arg3, %arg2, %arg1) : (tensor<1x224x224x3x!eltwise.fp32>, !int, !int, !int, !int) -> !tile.imap
      %6 = "tile.src_idx_map"(%0) : (tensor<!eltwise.fp32>) -> !tile.imap
      %7 = "tile.src_idx_map"(%2, %arg4, %arg3, %arg2, %arg1) : (tensor<1x224x224x3x!eltwise.int>, !int, !int, !int, !int) -> !tile.imap
      %8 = "tile.sink_idx_map"() : () -> !tile.imap
      %9 = "tile.size_map"() : () -> !tile.smap
      "tile.>(x==y?z)"(%9, %5, %6, %7, %8) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3"]} : () -> tensor<!eltwise.fp32>
    %4 = "eltwise.cast"(%3) : (tensor<!eltwise.fp32>) -> tensor<!eltwise.u32>
    return %4 : tensor<!eltwise.u32>
  }
}
)#"));
}

TEST(Op, BinaryCrossentropy) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, "I");
  auto O = Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, "O");
  Program program("binary_crossentropy", {op::binary_crossentropy(I, O, 0.0)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!float = type !eltwise.float
!int = type !eltwise.int
!fp32 = type !eltwise.fp32
!bool = type !eltwise.bool
module {
  func @binary_crossentropy(%arg0: tensor<7x7x3x64x!eltwise.fp32> {tile.name = "O"}, %arg1: tensor<7x7x3x64x!eltwise.fp32> {tile.name = "I"}) -> tensor<7x7x3x64x!eltwise.fp32> {
    %cst = "eltwise.sconst"() {value = 1.000000e+00 : f64} : () -> !float
    %cst_0 = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !float
    %c1 = "eltwise.sconst"() {value = 1 : i64} : () -> !int
    %0 = "eltwise.cmp_gt"(%arg0, %cst_0) {type = !fp32} : (tensor<7x7x3x64x!eltwise.fp32>, !float) -> tensor<7x7x3x64x!eltwise.bool>
    %1 = "eltwise.select"(%0, %arg0, %cst_0) {type = !fp32} : (tensor<7x7x3x64x!eltwise.bool>, tensor<7x7x3x64x!eltwise.fp32>, !float) -> tensor<7x7x3x64x!eltwise.fp32>
    %2 = "eltwise.cmp_lt"(%1, %cst) {type = !fp32} : (tensor<7x7x3x64x!eltwise.fp32>, !float) -> tensor<7x7x3x64x!eltwise.bool>
    %3 = "eltwise.select"(%2, %1, %cst) {type = !fp32} : (tensor<7x7x3x64x!eltwise.bool>, tensor<7x7x3x64x!eltwise.fp32>, !float) -> tensor<7x7x3x64x!eltwise.fp32>
    %4 = "eltwise.sub"(%c1, %3) {type = !fp32} : (!int, tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.fp32>
    %5 = "eltwise.log"(%4) {type = !fp32} : (tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.fp32>
    %6 = "eltwise.ident"(%arg1) {type = !fp32} : (tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.fp32>
    %7 = "eltwise.sub"(%c1, %6) {type = !fp32} : (!int, tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.fp32>
    %8 = "eltwise.mul"(%7, %5) {type = !fp32} : (tensor<7x7x3x64x!eltwise.fp32>, tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.fp32>
    %9 = "eltwise.log"(%3) {type = !fp32} : (tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.fp32>
    %10 = "eltwise.neg"(%6) {type = !fp32} : (tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.fp32>
    %11 = "eltwise.mul"(%10, %9) {type = !fp32} : (tensor<7x7x3x64x!eltwise.fp32>, tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.fp32>
    %12 = "eltwise.sub"(%11, %8) {type = !fp32} : (tensor<7x7x3x64x!eltwise.fp32>, tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.fp32>
    return %12 : tensor<7x7x3x64x!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, Clip) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, "I");
  auto raw_min = Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, "raw_min");
  auto raw_max = Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, "raw_max");
  Program program("clip", {op::clip(I, raw_min, raw_max)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!fp32 = type !eltwise.fp32
!bool = type !eltwise.bool
module {
  func @clip(%arg0: tensor<7x7x3x64x!eltwise.fp32> {tile.name = "raw_max"}, %arg1: tensor<7x7x3x64x!eltwise.fp32> {tile.name = "raw_min"}, %arg2: tensor<7x7x3x64x!eltwise.fp32> {tile.name = "I"}) -> tensor<7x7x3x64x!eltwise.fp32> {
    %0 = "eltwise.cmp_gt"(%arg2, %arg1) {type = !fp32} : (tensor<7x7x3x64x!eltwise.fp32>, tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.bool>
    %1 = "eltwise.select"(%0, %arg2, %arg1) {type = !fp32} : (tensor<7x7x3x64x!eltwise.bool>, tensor<7x7x3x64x!eltwise.fp32>, tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.fp32>
    %2 = "eltwise.cmp_lt"(%1, %arg0) {type = !fp32} : (tensor<7x7x3x64x!eltwise.fp32>, tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.bool>
    %3 = "eltwise.select"(%2, %1, %arg0) {type = !fp32} : (tensor<7x7x3x64x!eltwise.bool>, tensor<7x7x3x64x!eltwise.fp32>, tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.fp32>
    return %3 : tensor<7x7x3x64x!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, Concatenate) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, "A");
  auto B = Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, "B");
  Program program("concatenate", {op::concatenate({A, B}, 2)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!int = type !eltwise.int
!fp32 = type !eltwise.fp32
module {
  func @concatenate(%arg0: tensor<7x7x3x64x!eltwise.fp32> {tile.name = "B"}, %arg1: tensor<7x7x3x64x!eltwise.fp32> {tile.name = "A"}) -> tensor<7x7x6x64x!eltwise.fp32> {
    %c6 = "tile.affine_const"() {value = 6 : i64} : () -> !int
    %c64 = "tile.affine_const"() {value = 64 : i64} : () -> !int
    %c3 = "tile.affine_const"() {value = 3 : i64} : () -> !int
    %c7 = "tile.affine_const"() {value = 7 : i64} : () -> !int
    %0 = "tile.domain"() ( {
    ^bb0(%arg2: !int, %arg3: !int, %arg4: !int, %arg5: !int):	// no predecessors
      %3 = "tile.src_idx_map"(%arg0, %arg5, %arg4, %arg3, %arg2) : (tensor<7x7x3x64x!eltwise.fp32>, !int, !int, !int, !int) -> !tile.imap
      %4 = "tile.affine_add"(%arg3, %c3) : (!int, !int) -> !int
      %5 = "tile.sink_idx_map"(%arg5, %arg4, %4, %arg2) : (!int, !int, !int, !int) -> !tile.imap
      %6 = "tile.size_map"(%c7, %c7, %c6, %c64) : (!int, !int, !int, !int) -> !tile.smap
      "tile.=(x)"(%6, %3, %5) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["n3", "a", "n1", "n0"]} : () -> tensor<7x7x6x64x!eltwise.fp32>
    %1 = "tile.domain"() ( {
    ^bb0(%arg2: !int, %arg3: !int, %arg4: !int, %arg5: !int):	// no predecessors
      %3 = "tile.src_idx_map"(%arg1, %arg5, %arg4, %arg3, %arg2) : (tensor<7x7x3x64x!eltwise.fp32>, !int, !int, !int, !int) -> !tile.imap
      %4 = "tile.sink_idx_map"(%arg5, %arg4, %arg3, %arg2) : (!int, !int, !int, !int) -> !tile.imap
      %5 = "tile.size_map"(%c7, %c7, %c6, %c64) : (!int, !int, !int, !int) -> !tile.smap
      "tile.=(x)"(%5, %3, %4) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["n3", "a", "n1", "n0"]} : () -> tensor<7x7x6x64x!eltwise.fp32>
    %2 = "eltwise.add"(%1, %0) {type = !fp32} : (tensor<7x7x6x64x!eltwise.fp32>, tensor<7x7x6x64x!eltwise.fp32>) -> tensor<7x7x6x64x!eltwise.fp32>
    return %2 : tensor<7x7x6x64x!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, Convolution) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {1, 224, 224, 3}, "I");
  auto K = Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, "K");
  auto O = op::convolution(  //
      I,                     // I_or_O
      K,                     // F_or_O
      {2, 2},                // strides
      {1, 1},                // dilations
      {1, 1},                // data_dilations
      {},                    // filter_shape
      1,                     // groups
      "explicit",            // autopad_mode
      {3, 3},                // manual_padding
      "nxc",                 // input_layout
      "xck",                 // filter_layout
      "none",                // group_layout
      false,                 // winograd_allowed
      "",                    // name
      "ungrouped",           // autogroup_mode
      "none",                // deriv_mode
      {});                   // result_shape
  Program program("convolution", {O});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!int = type !eltwise.int
!fp32 = type !eltwise.fp32
module {
  func @convolution(%arg0: tensor<1x224x224x3x!eltwise.fp32> {tile.name = "I"}, %arg1: tensor<7x7x3x64x!eltwise.fp32> {tile.name = "K"}) -> tensor<1x112x112x64x!eltwise.fp32> {
    %c2 = "tile.affine_const"() {value = 2 : i64} : () -> !int
    %c3 = "tile.affine_const"() {value = 3 : i64} : () -> !int
    %c64 = "tile.affine_const"() {value = 64 : i64} : () -> !int
    %c112 = "tile.affine_const"() {value = 112 : i64} : () -> !int
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> !int
    %0 = "tile.domain"() ( {
    ^bb0(%arg2: !int, %arg3: !int, %arg4: !int, %arg5: !int, %arg6: !int, %arg7: !int, %arg8: !int):	// no predecessors
      %1 = "tile.affine_mul"(%arg4, %c2) : (!int, !int) -> !int
      %2 = "tile.affine_add"(%1, %arg3) : (!int, !int) -> !int
      %3 = "tile.affine_sub"(%2, %c3) : (!int, !int) -> !int
      %4 = "tile.affine_mul"(%arg6, %c2) : (!int, !int) -> !int
      %5 = "tile.affine_add"(%4, %arg5) : (!int, !int) -> !int
      %6 = "tile.affine_sub"(%5, %c3) : (!int, !int) -> !int
      %7 = "tile.src_idx_map"(%arg0, %arg7, %6, %3, %arg2) : (tensor<1x224x224x3x!eltwise.fp32>, !int, !int, !int, !int) -> !tile.imap
      %8 = "tile.src_idx_map"(%arg1, %arg5, %arg3, %arg2, %arg8) : (tensor<7x7x3x64x!eltwise.fp32>, !int, !int, !int, !int) -> !tile.imap
      %9 = "tile.sink_idx_map"(%arg7, %arg6, %arg4, %arg8) : (!int, !int, !int, !int) -> !tile.imap
      %10 = "tile.size_map"(%c1, %c112, %c112, %c64) : (!int, !int, !int, !int) -> !tile.smap
      "tile.+(x*y)"(%10, %7, %8, %9) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["ci", "k1", "x1", "k0", "x0", "n", "co"], name = "conv"} : () -> tensor<1x112x112x64x!eltwise.fp32>
    return %0 : tensor<1x112x112x64x!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, CumProd) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, "I");
  Program program("cumprod", {op::cumprod(I, 2)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!int = type !eltwise.int
!fp32 = type !eltwise.fp32
module {
  func @cumprod(%arg0: tensor<7x7x3x64x!eltwise.fp32> {tile.name = "I"}) -> tensor<7x7x3x64x!eltwise.fp32> {
    %c64 = "tile.affine_const"() {value = 64 : i64} : () -> !int
    %c7 = "tile.affine_const"() {value = 7 : i64} : () -> !int
    %c3 = "tile.affine_const"() {value = 3 : i64} : () -> !int
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !int, %arg2: !int, %arg3: !int, %arg4: !int, %arg5: !int):	// no predecessors
      %1 = "tile.affine_sub"(%arg3, %arg2) : (!int, !int) -> !int
      %2 = "tile.src_idx_map"(%arg0, %arg5, %arg4, %1, %arg1) : (tensor<7x7x3x64x!eltwise.fp32>, !int, !int, !int, !int) -> !tile.imap
      %3 = "tile.sink_idx_map"(%arg5, %arg4, %arg3, %arg1) : (!int, !int, !int, !int) -> !tile.imap
      %4 = "tile.size_map"(%c7, %c7, %c3, %c64) : (!int, !int, !int, !int) -> !tile.smap
      "tile.constraint"(%arg2, %c3) ( {
        "tile.*(x)"(%4, %2, %3) : (!tile.smap, !tile.imap, !tile.imap) -> ()
      }) : (!int, !int) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3", "x4"]} : () -> tensor<7x7x3x64x!eltwise.fp32>
    return %0 : tensor<7x7x3x64x!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, CumSum) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, "I");
  Program program("cumsum", {op::cumsum(I, 2)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!int = type !eltwise.int
!fp32 = type !eltwise.fp32
module {
  func @cumsum(%arg0: tensor<7x7x3x64x!eltwise.fp32> {tile.name = "I"}) -> tensor<7x7x3x64x!eltwise.fp32> {
    %c64 = "tile.affine_const"() {value = 64 : i64} : () -> !int
    %c7 = "tile.affine_const"() {value = 7 : i64} : () -> !int
    %c3 = "tile.affine_const"() {value = 3 : i64} : () -> !int
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !int, %arg2: !int, %arg3: !int, %arg4: !int, %arg5: !int):	// no predecessors
      %1 = "tile.affine_sub"(%arg3, %arg2) : (!int, !int) -> !int
      %2 = "tile.src_idx_map"(%arg0, %arg5, %arg4, %1, %arg1) : (tensor<7x7x3x64x!eltwise.fp32>, !int, !int, !int, !int) -> !tile.imap
      %3 = "tile.sink_idx_map"(%arg5, %arg4, %arg3, %arg1) : (!int, !int, !int, !int) -> !tile.imap
      %4 = "tile.size_map"(%c7, %c7, %c3, %c64) : (!int, !int, !int, !int) -> !tile.smap
      "tile.constraint"(%arg2, %c3) ( {
        "tile.+(x)"(%4, %2, %3) : (!tile.smap, !tile.imap, !tile.imap) -> ()
      }) : (!int, !int) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3", "x4"]} : () -> tensor<7x7x3x64x!eltwise.fp32>
    return %0 : tensor<7x7x3x64x!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, Dot) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, "I");
  auto K = Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, "K");
  Program program("dot", {op::dot(I, K)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!int = type !eltwise.int
!fp32 = type !eltwise.fp32
module {
  func @dot(%arg0: tensor<7x7x3x64x!eltwise.fp32> {tile.name = "I"}, %arg1: tensor<7x7x3x64x!eltwise.fp32> {tile.name = "K"}) -> tensor<7x7x3x7x7x64x!eltwise.fp32> {
    %c64 = "tile.affine_const"() {value = 64 : i64} : () -> !int
    %c3 = "tile.affine_const"() {value = 3 : i64} : () -> !int
    %c7 = "tile.affine_const"() {value = 7 : i64} : () -> !int
    %0 = "tile.domain"() ( {
    ^bb0(%arg2: !int, %arg3: !int, %arg4: !int, %arg5: !int, %arg6: !int, %arg7: !int, %arg8: !int):	// no predecessors
      %1 = "tile.src_idx_map"(%arg0, %arg5, %arg4, %arg3, %arg2) : (tensor<7x7x3x64x!eltwise.fp32>, !int, !int, !int, !int) -> !tile.imap
      %2 = "tile.src_idx_map"(%arg1, %arg8, %arg7, %arg2, %arg6) : (tensor<7x7x3x64x!eltwise.fp32>, !int, !int, !int, !int) -> !tile.imap
      %3 = "tile.sink_idx_map"(%arg5, %arg4, %arg3, %arg8, %arg7, %arg6) : (!int, !int, !int, !int, !int, !int) -> !tile.imap
      %4 = "tile.size_map"(%c7, %c7, %c3, %c7, %c7, %c64) : (!int, !int, !int, !int, !int, !int) -> !tile.smap
      "tile.+(x*y)"(%4, %1, %2, %3) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3", "x4", "x5", "x6"]} : () -> tensor<7x7x3x7x7x64x!eltwise.fp32>
    return %0 : tensor<7x7x3x7x7x64x!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, Elu) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, "I");
  Program program("elu", {op::elu(I, 0.1)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!float = type !eltwise.float
!int = type !eltwise.int
!fp32 = type !eltwise.fp32
!bool = type !eltwise.bool
module {
  func @elu(%arg0: tensor<7x7x3x64x!eltwise.fp32> {tile.name = "I"}) -> tensor<7x7x3x64x!eltwise.fp32> {
    %cst = "eltwise.sconst"() {value = 0.10000000149011612 : f64} : () -> !float
    %c0 = "eltwise.sconst"() {value = 0 : i64} : () -> !int
    %0 = "eltwise.exp"(%arg0) {type = !fp32} : (tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.fp32>
    %1 = "eltwise.mul"(%0, %cst) {type = !fp32} : (tensor<7x7x3x64x!eltwise.fp32>, !float) -> tensor<7x7x3x64x!eltwise.fp32>
    %2 = "eltwise.sub"(%1, %cst) {type = !fp32} : (tensor<7x7x3x64x!eltwise.fp32>, !float) -> tensor<7x7x3x64x!eltwise.fp32>
    %3 = "eltwise.cmp_lt"(%arg0, %c0) {type = !fp32} : (tensor<7x7x3x64x!eltwise.fp32>, !int) -> tensor<7x7x3x64x!eltwise.bool>
    %4 = "eltwise.select"(%3, %2, %arg0) {type = !fp32} : (tensor<7x7x3x64x!eltwise.bool>, tensor<7x7x3x64x!eltwise.fp32>, tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.fp32>
    return %4 : tensor<7x7x3x64x!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, ExpandDims) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, "I");
  Program program("expand_dims", {op::expand_dims(I, 2)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!int = type !eltwise.int
!fp32 = type !eltwise.fp32
module {
  func @expand_dims(%arg0: tensor<7x7x3x64x!eltwise.fp32> {tile.name = "I"}) -> tensor<7x7x1x3x64x!eltwise.fp32> {
    %c64 = "tile.affine_const"() {value = 64 : i64} : () -> !int
    %c3 = "tile.affine_const"() {value = 3 : i64} : () -> !int
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> !int
    %c7 = "tile.affine_const"() {value = 7 : i64} : () -> !int
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !int, %arg2: !int, %arg3: !int, %arg4: !int, %arg5: !int):	// no predecessors
      %1 = "tile.src_idx_map"(%arg0, %arg4, %arg3, %arg2, %arg1) : (tensor<7x7x3x64x!eltwise.fp32>, !int, !int, !int, !int) -> !tile.imap
      %2 = "tile.sink_idx_map"(%arg4, %arg3, %arg5, %arg2, %arg1) : (!int, !int, !int, !int, !int) -> !tile.imap
      %3 = "tile.size_map"(%c7, %c7, %c1, %c3, %c64) : (!int, !int, !int, !int, !int) -> !tile.smap
      "tile.=(x)"(%3, %1, %2) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["n3", "n2", "n1", "n0", "a"]} : () -> tensor<7x7x1x3x64x!eltwise.fp32>
    return %0 : tensor<7x7x1x3x64x!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, Flip) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, "I");
  Program program("flip", {op::flip(I, 2)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!int = type !eltwise.int
!fp32 = type !eltwise.fp32
module {
  func @flip(%arg0: tensor<7x7x3x64x!eltwise.fp32> {tile.name = "I"}) -> tensor<7x7x3x64x!eltwise.fp32> {
    %c2 = "tile.affine_const"() {value = 2 : i64} : () -> !int
    %c3 = "tile.affine_const"() {value = 3 : i64} : () -> !int
    %c64 = "tile.affine_const"() {value = 64 : i64} : () -> !int
    %c7 = "tile.affine_const"() {value = 7 : i64} : () -> !int
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !int, %arg2: !int, %arg3: !int, %arg4: !int):	// no predecessors
      %1 = "tile.src_idx_map"(%arg0, %arg4, %arg3, %arg2, %arg1) : (tensor<7x7x3x64x!eltwise.fp32>, !int, !int, !int, !int) -> !tile.imap
      %2 = "tile.affine_sub"(%c2, %arg2) : (!int, !int) -> !int
      %3 = "tile.sink_idx_map"(%arg4, %arg3, %2, %arg1) : (!int, !int, !int, !int) -> !tile.imap
      %4 = "tile.size_map"(%c7, %c7, %c3, %c64) : (!int, !int, !int, !int) -> !tile.smap
      "tile.=(x)"(%4, %1, %3) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3"]} : () -> tensor<7x7x3x64x!eltwise.fp32>
    return %0 : tensor<7x7x3x64x!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, HardSigmoid) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  Program program("hard_sigmoid", {op::hard_sigmoid(A, 0.05)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!float = type !eltwise.float
!fp32 = type !eltwise.fp32
!bool = type !eltwise.bool
module {
  func @hard_sigmoid(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "A"}) -> tensor<10x20x!eltwise.fp32> {
    %cst = "eltwise.sconst"() {value = 5.000000e-01 : f64} : () -> !float
    %cst_0 = "eltwise.sconst"() {value = 0.05000000074505806 : f64} : () -> !float
    %cst_1 = "eltwise.sconst"() {value = 1.000000e+00 : f64} : () -> !float
    %cst_2 = "eltwise.sconst"() {value = 9.9999998509883898 : f64} : () -> !float
    %cst_3 = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !float
    %cst_4 = "eltwise.sconst"() {value = -9.9999998509883898 : f64} : () -> !float
    %0 = "eltwise.mul"(%arg0, %cst_0) {type = !fp32} : (tensor<10x20x!eltwise.fp32>, !float) -> tensor<10x20x!eltwise.fp32>
    %1 = "eltwise.add"(%0, %cst) {type = !fp32} : (tensor<10x20x!eltwise.fp32>, !float) -> tensor<10x20x!eltwise.fp32>
    %2 = "eltwise.cmp_gt"(%arg0, %cst_2) {type = !fp32} : (tensor<10x20x!eltwise.fp32>, !float) -> tensor<10x20x!eltwise.bool>
    %3 = "eltwise.select"(%2, %cst_1, %1) {type = !fp32} : (tensor<10x20x!eltwise.bool>, !float, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    %4 = "eltwise.cmp_lt"(%arg0, %cst_4) {type = !fp32} : (tensor<10x20x!eltwise.fp32>, !float) -> tensor<10x20x!eltwise.bool>
    %5 = "eltwise.select"(%4, %cst_3, %3) {type = !fp32} : (tensor<10x20x!eltwise.bool>, !float, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    return %5 : tensor<10x20x!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, ImageResize) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {1, 224, 224, 3}, "I");
  auto image_resize = op::image_resize(I, std::vector<int>{5, 4}, "bilinear", "nxc");
  Program program("image_resize", {image_resize});
  IVLOG(1, program);
  //   EXPECT_THAT(program, Eq(R"(function (
  //   I[I_0, I_1, I_2, I_3]
  // ) -> (
  //   _X7
  // ) {
  //   _X0 = 0.200000;
  //   _X1[y : 5] = =(_X0[]);
  //   _X2[y : 9] = +(_X1[-4 + j + y]), j < 5;
  //   _X3 = 0.250000;
  //   _X4[x : 4] = =(_X3[]);
  //   _X5[x : 7] = +(_X4[-3 + i + x]), i < 4;
  //   _X6[y, x : 9, 7] = =(_X2[y] * _X5[x]);
  //   _X7[x0, -4 + j + 5*x1, -3 + i + 4*x2, x3 : 1, 1120, 896, 3] = +(I[x0, x1, x2, x3] * _X6[j, i]);
  // }
  // )"));
}

TEST(Op, Max) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {1, 224, 224, 3}, "I");
  Program program("max", {op::max(I)});  // NOLINT(build/include_what_you_use)
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!fp32 = type !eltwise.fp32
!int = type !eltwise.int
module {
  func @max(%arg0: tensor<1x224x224x3x!eltwise.fp32> {tile.name = "I"}) -> tensor<!eltwise.fp32> {
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !int, %arg2: !int, %arg3: !int, %arg4: !int):	// no predecessors
      %1 = "tile.src_idx_map"(%arg0, %arg4, %arg3, %arg2, %arg1) : (tensor<1x224x224x3x!eltwise.fp32>, !int, !int, !int, !int) -> !tile.imap
      %2 = "tile.sink_idx_map"() : () -> !tile.imap
      %3 = "tile.size_map"() : () -> !tile.smap
      "tile.>(x)"(%3, %1, %2) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3"]} : () -> tensor<!eltwise.fp32>
    return %0 : tensor<!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, Maximum) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  auto B = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "B");
  Program program("maximum", {op::maximum(A, B)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!fp32 = type !eltwise.fp32
!bool = type !eltwise.bool
module {
  func @maximum(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "A"}, %arg1: tensor<10x20x!eltwise.fp32> {tile.name = "B"}) -> tensor<10x20x!eltwise.fp32> {
    %0 = "eltwise.cmp_lt"(%arg0, %arg1) {type = !fp32} : (tensor<10x20x!eltwise.fp32>, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.bool>
    %1 = "eltwise.select"(%0, %arg1, %arg0) {type = !fp32} : (tensor<10x20x!eltwise.bool>, tensor<10x20x!eltwise.fp32>, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    return %1 : tensor<10x20x!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, Mean) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  Program program("mean", {op::mean(A)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!int = type !eltwise.int
!fp32 = type !eltwise.fp32
module {
  func @mean(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "A"}) -> !fp32 {
    %c200 = "eltwise.sconst"() {value = 200 : i64} : () -> !int
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !int, %arg2: !int):	// no predecessors
      %2 = "tile.src_idx_map"(%arg0, %arg2, %arg1) : (tensor<10x20x!eltwise.fp32>, !int, !int) -> !tile.imap
      %3 = "tile.sink_idx_map"() : () -> !tile.imap
      %4 = "tile.size_map"() : () -> !tile.smap
      "tile.+(x)"(%4, %2, %3) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1"]} : () -> tensor<!eltwise.fp32>
    %1 = "eltwise.div"(%0, %c200) {type = !fp32} : (tensor<!eltwise.fp32>, !int) -> !fp32
    return %1 : !fp32
  }
}
)#"));
}

TEST(Op, Min) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  Program program("min", {op::min(A)});  // NOLINT(build/include_what_you_use)
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!fp32 = type !eltwise.fp32
!int = type !eltwise.int
module {
  func @min(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "A"}) -> tensor<!eltwise.fp32> {
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !int, %arg2: !int):	// no predecessors
      %1 = "tile.src_idx_map"(%arg0, %arg2, %arg1) : (tensor<10x20x!eltwise.fp32>, !int, !int) -> !tile.imap
      %2 = "tile.sink_idx_map"() : () -> !tile.imap
      %3 = "tile.size_map"() : () -> !tile.smap
      "tile.<(x)"(%3, %1, %2) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1"]} : () -> tensor<!eltwise.fp32>
    return %0 : tensor<!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, Minimum) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  auto B = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "B");
  Program program("minimum", {op::minimum(A, B)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!fp32 = type !eltwise.fp32
!bool = type !eltwise.bool
module {
  func @minimum(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "B"}, %arg1: tensor<10x20x!eltwise.fp32> {tile.name = "A"}) -> tensor<10x20x!eltwise.fp32> {
    %0 = "eltwise.cmp_lt"(%arg1, %arg0) {type = !fp32} : (tensor<10x20x!eltwise.fp32>, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.bool>
    %1 = "eltwise.select"(%0, %arg1, %arg0) {type = !fp32} : (tensor<10x20x!eltwise.bool>, tensor<10x20x!eltwise.fp32>, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    return %1 : tensor<10x20x!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, Pool) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20, 30, 40, 50}, "I");
  Program program("pool", {op::pool(I, "sum", {1, 2, 3}, {1, 2, 3}, "none", {1, 2}, "nwc", true, true)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!int = type !eltwise.int
!fp32 = type !eltwise.fp32
module {
  func @pool(%arg0: tensor<10x20x30x40x50x!eltwise.fp32> {tile.name = "I"}) -> tensor<10x22x17x14x50x!eltwise.fp32> {
    %c50 = "tile.affine_const"() {value = 50 : i64} : () -> !int
    %c14 = "tile.affine_const"() {value = 14 : i64} : () -> !int
    %c17 = "tile.affine_const"() {value = 17 : i64} : () -> !int
    %c22 = "tile.affine_const"() {value = 22 : i64} : () -> !int
    %c10 = "tile.affine_const"() {value = 10 : i64} : () -> !int
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> !int
    %c2 = "tile.affine_const"() {value = 2 : i64} : () -> !int
    %c3 = "tile.affine_const"() {value = 3 : i64} : () -> !int
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !int, %arg2: !int, %arg3: !int, %arg4: !int, %arg5: !int, %arg6: !int, %arg7: !int, %arg8: !int):	// no predecessors
      %1 = "tile.affine_mul"(%arg3, %c3) : (!int, !int) -> !int
      %2 = "tile.affine_add"(%1, %arg2) : (!int, !int) -> !int
      %3 = "tile.affine_mul"(%arg5, %c2) : (!int, !int) -> !int
      %4 = "tile.affine_add"(%3, %arg4) : (!int, !int) -> !int
      %5 = "tile.affine_sub"(%4, %c2) : (!int, !int) -> !int
      %6 = "tile.affine_add"(%arg7, %arg6) : (!int, !int) -> !int
      %7 = "tile.affine_sub"(%6, %c1) : (!int, !int) -> !int
      %8 = "tile.src_idx_map"(%arg0, %arg8, %7, %5, %2, %arg1) : (tensor<10x20x30x40x50x!eltwise.fp32>, !int, !int, !int, !int, !int) -> !tile.imap
      %9 = "tile.sink_idx_map"(%arg8, %arg7, %arg5, %arg3, %arg1) : (!int, !int, !int, !int, !int) -> !tile.imap
      %10 = "tile.size_map"(%c10, %c22, %c17, %c14, %c50) : (!int, !int, !int, !int, !int) -> !tile.smap
      "tile.constraint"(%arg2, %c3) ( {
        "tile.constraint"(%arg4, %c2) ( {
          "tile.constraint"(%arg6, %c1) ( {
            "tile.+(x)"(%10, %8, %9) : (!tile.smap, !tile.imap, !tile.imap) -> ()
          }) : (!int, !int) -> ()
        }) : (!int, !int) -> ()
      }) : (!int, !int) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"]} : () -> tensor<10x22x17x14x50x!eltwise.fp32>
    return %0 : tensor<10x22x17x14x50x!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, Prod) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  Program program("prod", {op::prod(A)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!fp32 = type !eltwise.fp32
!int = type !eltwise.int
module {
  func @prod(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "A"}) -> tensor<!eltwise.fp32> {
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !int, %arg2: !int):	// no predecessors
      %1 = "tile.src_idx_map"(%arg0, %arg2, %arg1) : (tensor<10x20x!eltwise.fp32>, !int, !int) -> !tile.imap
      %2 = "tile.sink_idx_map"() : () -> !tile.imap
      %3 = "tile.size_map"() : () -> !tile.smap
      "tile.*(x)"(%3, %1, %2) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1"]} : () -> tensor<!eltwise.fp32>
    return %0 : tensor<!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, Relu) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "I");
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  auto M = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "M");
  Program program("relu", {op::relu(I).alpha(A).max_value(M).threshold(0.05)});
  EXPECT_THAT(program, Eq(R"#(

!float = type !eltwise.float
!fp32 = type !eltwise.fp32
!bool = type !eltwise.bool
module {
  func @relu(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "M"}, %arg1: tensor<10x20x!eltwise.fp32> {tile.name = "I"}, %arg2: tensor<10x20x!eltwise.fp32> {tile.name = "A"}) -> tensor<10x20x!eltwise.fp32> {
    %cst = "eltwise.sconst"() {value = 0.05000000074505806 : f64} : () -> !float
    %0 = "eltwise.sub"(%arg1, %cst) {type = !fp32} : (tensor<10x20x!eltwise.fp32>, !float) -> tensor<10x20x!eltwise.fp32>
    %1 = "eltwise.mul"(%arg2, %0) {type = !fp32} : (tensor<10x20x!eltwise.fp32>, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    %2 = "eltwise.cmp_lt"(%arg1, %cst) {type = !fp32} : (tensor<10x20x!eltwise.fp32>, !float) -> tensor<10x20x!eltwise.bool>
    %3 = "eltwise.select"(%2, %1, %arg1) {type = !fp32} : (tensor<10x20x!eltwise.bool>, tensor<10x20x!eltwise.fp32>, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    %4 = "eltwise.cmp_lt"(%3, %arg0) {type = !fp32} : (tensor<10x20x!eltwise.fp32>, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.bool>
    %5 = "eltwise.select"(%4, %3, %arg0) {type = !fp32} : (tensor<10x20x!eltwise.bool>, tensor<10x20x!eltwise.fp32>, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    return %5 : tensor<10x20x!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, ReluNoAlpha) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "I");
  auto M = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "M");
  Program program("relu", {op::relu(I).max_value(M).threshold(0.05)});
  EXPECT_THAT(program, Eq(R"#(

!float = type !eltwise.float
!fp32 = type !eltwise.fp32
!bool = type !eltwise.bool
module {
  func @relu(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "M"}, %arg1: tensor<10x20x!eltwise.fp32> {tile.name = "I"}) -> tensor<10x20x!eltwise.fp32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !float
    %cst_0 = "eltwise.sconst"() {value = 0.05000000074505806 : f64} : () -> !float
    %0 = "eltwise.cmp_lt"(%arg1, %cst_0) {type = !fp32} : (tensor<10x20x!eltwise.fp32>, !float) -> tensor<10x20x!eltwise.bool>
    %1 = "eltwise.select"(%0, %cst, %arg1) {type = !fp32} : (tensor<10x20x!eltwise.bool>, !float, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    %2 = "eltwise.cmp_lt"(%1, %arg0) {type = !fp32} : (tensor<10x20x!eltwise.fp32>, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.bool>
    %3 = "eltwise.select"(%2, %1, %arg0) {type = !fp32} : (tensor<10x20x!eltwise.bool>, tensor<10x20x!eltwise.fp32>, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    return %3 : tensor<10x20x!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, ReluNoMaxValue) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "I");
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  Program program("relu", {op::relu(I).alpha(A).threshold(0.05)});
  EXPECT_THAT(program, Eq(R"#(

!float = type !eltwise.float
!fp32 = type !eltwise.fp32
!bool = type !eltwise.bool
module {
  func @relu(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "I"}, %arg1: tensor<10x20x!eltwise.fp32> {tile.name = "A"}) -> tensor<10x20x!eltwise.fp32> {
    %cst = "eltwise.sconst"() {value = 0.05000000074505806 : f64} : () -> !float
    %0 = "eltwise.sub"(%arg0, %cst) {type = !fp32} : (tensor<10x20x!eltwise.fp32>, !float) -> tensor<10x20x!eltwise.fp32>
    %1 = "eltwise.mul"(%arg1, %0) {type = !fp32} : (tensor<10x20x!eltwise.fp32>, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    %2 = "eltwise.cmp_lt"(%arg0, %cst) {type = !fp32} : (tensor<10x20x!eltwise.fp32>, !float) -> tensor<10x20x!eltwise.bool>
    %3 = "eltwise.select"(%2, %1, %arg0) {type = !fp32} : (tensor<10x20x!eltwise.bool>, tensor<10x20x!eltwise.fp32>, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    return %3 : tensor<10x20x!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, ReluOnlyThreshold) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "I");
  Program program("relu", {op::relu(I).threshold(0.05)});
  EXPECT_THAT(program, Eq(R"#(

!float = type !eltwise.float
!fp32 = type !eltwise.fp32
!bool = type !eltwise.bool
module {
  func @relu(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "I"}) -> tensor<10x20x!eltwise.fp32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !float
    %cst_0 = "eltwise.sconst"() {value = 0.05000000074505806 : f64} : () -> !float
    %0 = "eltwise.cmp_lt"(%arg0, %cst_0) {type = !fp32} : (tensor<10x20x!eltwise.fp32>, !float) -> tensor<10x20x!eltwise.bool>
    %1 = "eltwise.select"(%0, %cst, %arg0) {type = !fp32} : (tensor<10x20x!eltwise.bool>, !float, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    return %1 : tensor<10x20x!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, ReluNoParams) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "I");
  Program program("relu", {op::relu(I)});
  EXPECT_THAT(program, Eq(R"#(

!float = type !eltwise.float
!fp32 = type !eltwise.fp32
!bool = type !eltwise.bool
module {
  func @relu(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "I"}) -> tensor<10x20x!eltwise.fp32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !float
    %0 = "eltwise.cmp_lt"(%arg0, %cst) {type = !fp32} : (tensor<10x20x!eltwise.fp32>, !float) -> tensor<10x20x!eltwise.bool>
    %1 = "eltwise.select"(%0, %cst, %arg0) {type = !fp32} : (tensor<10x20x!eltwise.bool>, !float, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    return %1 : tensor<10x20x!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, Repeat) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {32, 1, 4, 1}, "A");
  auto X = op::repeat(  //
      A,                // tensor to repeat
      3,                // number of repeats
      2);               // axis to repeat
  Program program("repeat", {X});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!int = type !eltwise.int
!fp32 = type !eltwise.fp32
module {
  func @repeat(%arg0: tensor<32x1x4x1x!eltwise.fp32> {tile.name = "A"}) -> tensor<32x1x12x1x!eltwise.fp32> {
    %c12 = "tile.affine_const"() {value = 12 : i64} : () -> !int
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> !int
    %c32 = "tile.affine_const"() {value = 32 : i64} : () -> !int
    %c3 = "tile.affine_const"() {value = 3 : i64} : () -> !int
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !int, %arg2: !int, %arg3: !int, %arg4: !int, %arg5: !int):	// no predecessors
      %1 = "tile.src_idx_map"(%arg0, %arg4, %arg3, %arg2, %arg1) : (tensor<32x1x4x1x!eltwise.fp32>, !int, !int, !int, !int) -> !tile.imap
      %2 = "tile.affine_mul"(%arg2, %c3) : (!int, !int) -> !int
      %3 = "tile.affine_add"(%2, %arg5) : (!int, !int) -> !int
      %4 = "tile.sink_idx_map"(%arg4, %arg3, %3, %arg1) : (!int, !int, !int, !int) -> !tile.imap
      %5 = "tile.size_map"(%c32, %c1, %c12, %c1) : (!int, !int, !int, !int) -> !tile.smap
      "tile.constraint"(%arg5, %c3) ( {
        "tile.=(x)"(%5, %1, %4) : (!tile.smap, !tile.imap, !tile.imap) -> ()
      }) : (!int, !int) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3", "x4"]} : () -> tensor<32x1x12x1x!eltwise.fp32>
    return %0 : tensor<32x1x12x1x!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, Reshape) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  TensorDim I, J;
  A.bind_dims(I, J);
  Program program("reshape", {op::reshape(A, make_tuple(J, I))});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!int = type !eltwise.int
!fp32 = type !eltwise.fp32
module {
  func @reshape(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "A"}) -> tensor<20x10x!eltwise.fp32> {
    %c10 = "tile.affine_const"() {value = 10 : i64} : () -> !int
    %c20 = "tile.affine_const"() {value = 20 : i64} : () -> !int
    %0 = "tile.reshape"(%arg0, %c20, %c10) : (tensor<10x20x!eltwise.fp32>, !int, !int) -> tensor<20x10x!eltwise.fp32>
    return %0 : tensor<20x10x!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, Sigmoid) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10}, "A");
  Program program("sigmoid", {op::sigmoid(A)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!float = type !eltwise.float
!fp32 = type !eltwise.fp32
module {
  func @sigmoid(%arg0: tensor<10x!eltwise.fp32> {tile.name = "A"}) -> tensor<10x!eltwise.fp32> {
    %cst = "eltwise.sconst"() {value = 1.000000e+00 : f64} : () -> !float
    %0 = "eltwise.ident"(%arg0) {type = !fp32} : (tensor<10x!eltwise.fp32>) -> tensor<10x!eltwise.fp32>
    %1 = "eltwise.neg"(%0) {type = !fp32} : (tensor<10x!eltwise.fp32>) -> tensor<10x!eltwise.fp32>
    %2 = "eltwise.exp"(%1) {type = !fp32} : (tensor<10x!eltwise.fp32>) -> tensor<10x!eltwise.fp32>
    %3 = "eltwise.add"(%2, %cst) {type = !fp32} : (tensor<10x!eltwise.fp32>, !float) -> tensor<10x!eltwise.fp32>
    %4 = "eltwise.div"(%cst, %3) {type = !fp32} : (!float, tensor<10x!eltwise.fp32>) -> tensor<10x!eltwise.fp32>
    return %4 : tensor<10x!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, Slice) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  auto X = op::slice(  //
      A,               // tensor to perform spatial padding on
      {2, 10});        // slices
  Program program("slice", {X});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!int = type !eltwise.int
!fp32 = type !eltwise.fp32
module {
  func @slice(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "A"}) -> tensor<!eltwise.fp32> {
    %c10 = "tile.affine_const"() {value = 10 : i64} : () -> !int
    %c2 = "tile.affine_const"() {value = 2 : i64} : () -> !int
    %0 = "tile.domain"() ( {
      %1 = "tile.src_idx_map"(%arg0, %c2, %c10) : (tensor<10x20x!eltwise.fp32>, !int, !int) -> !tile.imap
      %2 = "tile.sink_idx_map"() : () -> !tile.imap
      %3 = "tile.size_map"() : () -> !tile.smap
      "tile.=(x)"(%3, %1, %2) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = []} : () -> tensor<!eltwise.fp32>
    return %0 : tensor<!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, Softmax) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  Program program("softmax", {op::softmax(A, 1)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!int = type !eltwise.int
!fp32 = type !eltwise.fp32
module {
  func @softmax(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "A"}) -> tensor<10x20x!eltwise.fp32> {
    %c0 = "tile.affine_const"() {value = 0 : i64} : () -> !int
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> !int
    %c10 = "tile.affine_const"() {value = 10 : i64} : () -> !int
    %0 = "eltwise.ident"(%arg0) {type = !fp32} : (tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    %1 = "tile.domain"() ( {
    ^bb0(%arg1: !int, %arg2: !int):	// no predecessors
      %6 = "tile.src_idx_map"(%0, %arg2, %arg1) : (tensor<10x20x!eltwise.fp32>, !int, !int) -> !tile.imap
      %7 = "tile.sink_idx_map"(%arg2, %c0) : (!int, !int) -> !tile.imap
      %8 = "tile.size_map"(%c10, %c1) : (!int, !int) -> !tile.smap
      "tile.>(x)"(%8, %6, %7) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1"]} : () -> tensor<10x1x!eltwise.fp32>
    %2 = "eltwise.sub"(%0, %1) {type = !fp32} : (tensor<10x20x!eltwise.fp32>, tensor<10x1x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    %3 = "eltwise.exp"(%2) {type = !fp32} : (tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    %4 = "tile.domain"() ( {
    ^bb0(%arg1: !int, %arg2: !int):	// no predecessors
      %6 = "tile.src_idx_map"(%3, %arg2, %arg1) : (tensor<10x20x!eltwise.fp32>, !int, !int) -> !tile.imap
      %7 = "tile.sink_idx_map"(%arg2, %c0) : (!int, !int) -> !tile.imap
      %8 = "tile.size_map"(%c10, %c1) : (!int, !int) -> !tile.smap
      "tile.+(x)"(%8, %6, %7) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1"]} : () -> tensor<10x1x!eltwise.fp32>
    %5 = "eltwise.div"(%3, %4) {type = !fp32} : (tensor<10x20x!eltwise.fp32>, tensor<10x1x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    return %5 : tensor<10x20x!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, SpatialPadding) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {64, 4, 32, 32}, "A");
  auto X = op::spatial_padding(  //
      A,                         // tensor to perform spatial padding on
      {1, 3},                    // low pads
      {3, 3},                    // high pads
      "nchw");                   // data layout
  Program program("spatial_padding", {X});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!int = type !eltwise.int
!fp32 = type !eltwise.fp32
module {
  func @spatial_padding(%arg0: tensor<64x4x32x32x!eltwise.fp32> {tile.name = "A"}) -> tensor<64x4x36x38x!eltwise.fp32> {
    %c3 = "tile.affine_const"() {value = 3 : i64} : () -> !int
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> !int
    %c38 = "tile.affine_const"() {value = 38 : i64} : () -> !int
    %c36 = "tile.affine_const"() {value = 36 : i64} : () -> !int
    %c4 = "tile.affine_const"() {value = 4 : i64} : () -> !int
    %c64 = "tile.affine_const"() {value = 64 : i64} : () -> !int
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !int, %arg2: !int, %arg3: !int, %arg4: !int):	// no predecessors
      %1 = "tile.src_idx_map"(%arg0, %arg4, %arg3, %arg2, %arg1) : (tensor<64x4x32x32x!eltwise.fp32>, !int, !int, !int, !int) -> !tile.imap
      %2 = "tile.affine_add"(%arg1, %c3) : (!int, !int) -> !int
      %3 = "tile.affine_add"(%arg2, %c1) : (!int, !int) -> !int
      %4 = "tile.sink_idx_map"(%arg4, %arg3, %3, %2) : (!int, !int, !int, !int) -> !tile.imap
      %5 = "tile.size_map"(%c64, %c4, %c36, %c38) : (!int, !int, !int, !int) -> !tile.smap
      "tile.=(x)"(%5, %1, %4) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x1", "x0", "c", "n"]} : () -> tensor<64x4x36x38x!eltwise.fp32>
    return %0 : tensor<64x4x36x38x!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, Square) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10}, "A");
  Program program("square", {op::square(A)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!fp32 = type !eltwise.fp32
module {
  func @square(%arg0: tensor<10x!eltwise.fp32> {tile.name = "A"}) -> tensor<10x!eltwise.fp32> {
    %0 = "eltwise.mul"(%arg0, %arg0) {type = !fp32} : (tensor<10x!eltwise.fp32>, tensor<10x!eltwise.fp32>) -> tensor<10x!eltwise.fp32>
    return %0 : tensor<10x!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, Sum) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  Program program("sum", {op::sum(A)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!fp32 = type !eltwise.fp32
!int = type !eltwise.int
module {
  func @sum(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "A"}) -> tensor<!eltwise.fp32> {
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !int, %arg2: !int):	// no predecessors
      %1 = "tile.src_idx_map"(%arg0, %arg2, %arg1) : (tensor<10x20x!eltwise.fp32>, !int, !int) -> !tile.imap
      %2 = "tile.sink_idx_map"() : () -> !tile.imap
      %3 = "tile.size_map"() : () -> !tile.smap
      "tile.+(x)"(%3, %1, %2) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1"]} : () -> tensor<!eltwise.fp32>
    return %0 : tensor<!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, Squeeze) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {32, 1, 4, 1}, "A");
  auto X = op::squeeze(  //
      A,                 // tensor to squeeze
      {1, 3});           // axes to squeeze
  Program program("squeeze", {X});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!int = type !eltwise.int
!fp32 = type !eltwise.fp32
module {
  func @squeeze(%arg0: tensor<32x1x4x1x!eltwise.fp32> {tile.name = "A"}) -> tensor<32x4x!eltwise.fp32> {
    %c4 = "tile.affine_const"() {value = 4 : i64} : () -> !int
    %c32 = "tile.affine_const"() {value = 32 : i64} : () -> !int
    %0 = "tile.reshape"(%arg0, %c32, %c4) : (tensor<32x1x4x1x!eltwise.fp32>, !int, !int) -> tensor<32x4x!eltwise.fp32>
    return %0 : tensor<32x4x!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, Tile) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  auto X = op::tile(  //
      A,              // tensor to tile
      {5, 4});        // tiling factors
  Program program("tile", {X});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!int = type !eltwise.int
!fp32 = type !eltwise.fp32
module {
  func @tile(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "A"}) -> tensor<50x80x!eltwise.fp32> {
    %c80 = "tile.affine_const"() {value = 80 : i64} : () -> !int
    %c20 = "tile.affine_const"() {value = 20 : i64} : () -> !int
    %c50 = "tile.affine_const"() {value = 50 : i64} : () -> !int
    %c10 = "tile.affine_const"() {value = 10 : i64} : () -> !int
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !int, %arg2: !int, %arg3: !int, %arg4: !int):	// no predecessors
      %1 = "tile.src_idx_map"(%arg0, %arg2, %arg1) : (tensor<10x20x!eltwise.fp32>, !int, !int) -> !tile.imap
      %2 = "tile.affine_mul"(%arg3, %c20) : (!int, !int) -> !int
      %3 = "tile.affine_add"(%2, %arg1) : (!int, !int) -> !int
      %4 = "tile.affine_mul"(%arg4, %c10) : (!int, !int) -> !int
      %5 = "tile.affine_add"(%4, %arg2) : (!int, !int) -> !int
      %6 = "tile.sink_idx_map"(%5, %3) : (!int, !int) -> !tile.imap
      %7 = "tile.size_map"(%c50, %c80) : (!int, !int) -> !tile.smap
      "tile.=(x)"(%7, %1, %6) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3"], no_reduce = true} : () -> tensor<50x80x!eltwise.fp32>
    return %0 : tensor<50x80x!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, Transpose) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  Program program("transpose", {op::transpose(A)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!int = type !eltwise.int
!fp32 = type !eltwise.fp32
module {
  func @transpose(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "A"}) -> tensor<20x10x!eltwise.fp32> {
    %c10 = "tile.affine_const"() {value = 10 : i64} : () -> !int
    %c20 = "tile.affine_const"() {value = 20 : i64} : () -> !int
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !int, %arg2: !int):	// no predecessors
      %1 = "tile.src_idx_map"(%arg0, %arg2, %arg1) : (tensor<10x20x!eltwise.fp32>, !int, !int) -> !tile.imap
      %2 = "tile.sink_idx_map"(%arg1, %arg2) : (!int, !int) -> !tile.imap
      %3 = "tile.size_map"(%c20, %c10) : (!int, !int) -> !tile.smap
      "tile.=(x)"(%3, %1, %2) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1"]} : () -> tensor<20x10x!eltwise.fp32>
    return %0 : tensor<20x10x!eltwise.fp32>
  }
}
)#"));
}

TEST(Op, Variance) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  Program program("variance", {op::variance(A)});
  IVLOG(1, program);
  EXPECT_THAT(program, Eq(R"#(

!int = type !eltwise.int
!fp32 = type !eltwise.fp32
module {
  func @variance(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "A"}) -> tensor<!eltwise.fp32> {
    %c200 = "eltwise.sconst"() {value = 200 : i64} : () -> !int
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> !int
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !int, %arg2: !int, %arg3: !int, %arg4: !int):	// no predecessors
      %6 = "tile.src_idx_map"(%arg0, %arg2, %arg1) : (tensor<10x20x!eltwise.fp32>, !int, !int) -> !tile.imap
      %7 = "tile.sink_idx_map"(%arg4, %arg3) : (!int, !int) -> !tile.imap
      %8 = "tile.size_map"(%c1, %c1) : (!int, !int) -> !tile.smap
      "tile.+(x)"(%8, %6, %7) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3"]} : () -> tensor<1x1x!eltwise.fp32>
    %1 = "eltwise.div"(%0, %c200) {type = !fp32} : (tensor<1x1x!eltwise.fp32>, !int) -> tensor<1x1x!eltwise.fp32>
    %2 = "eltwise.sub"(%arg0, %1) {type = !fp32} : (tensor<10x20x!eltwise.fp32>, tensor<1x1x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    %3 = "eltwise.mul"(%2, %2) {type = !fp32} : (tensor<10x20x!eltwise.fp32>, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    %4 = "tile.domain"() ( {
    ^bb0(%arg1: !int, %arg2: !int):	// no predecessors
      %6 = "tile.src_idx_map"(%3, %arg2, %arg1) : (tensor<10x20x!eltwise.fp32>, !int, !int) -> !tile.imap
      %7 = "tile.sink_idx_map"() : () -> !tile.imap
      %8 = "tile.size_map"() : () -> !tile.smap
      "tile.+(x)"(%8, %6, %7) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1"]} : () -> tensor<!eltwise.fp32>
    %5 = "eltwise.div"(%4, %c200) {type = !fp32} : (tensor<!eltwise.fp32>, !int) -> tensor<!eltwise.fp32>
    return %5 : tensor<!eltwise.fp32>
  }
}
)#"));
}

}  // namespace
}  // namespace plaidml::op
