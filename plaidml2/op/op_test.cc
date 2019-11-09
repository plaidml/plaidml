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

#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1, I_2, I_3]
) -> (
  _X3
) {
  _X0 = 0.000000;
  _X1 = cmp_lt(I, _X0);
  _X2 = neg(I);
  _X3 = cond(_X1, _X2, I);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

!float = type tensor<!eltwise.float>
module {
  func @abs(%arg0: tensor<1x224x224x3x!eltwise.fp32> {tile.name = "I"}) -> tensor<1x224x224x3x!eltwise.fp32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !float
    %0 = "eltwise.neg"(%arg0) {type = !eltwise.fp32} : (tensor<1x224x224x3x!eltwise.fp32>) -> tensor<1x224x224x3x!eltwise.fp32>
    %1 = "eltwise.cmp_lt"(%arg0, %cst) {type = !eltwise.fp32} : (tensor<1x224x224x3x!eltwise.fp32>, !float) -> tensor<1x224x224x3x!eltwise.bool>
    %2 = "eltwise.select"(%1, %0, %arg0) {type = !eltwise.fp32} : (tensor<1x224x224x3x!eltwise.bool>, tensor<1x224x224x3x!eltwise.fp32>, tensor<1x224x224x3x!eltwise.fp32>) -> tensor<1x224x224x3x!eltwise.fp32>
    return %2 : tensor<1x224x224x3x!eltwise.fp32>
  }
}
)#"));
#endif
}

TEST(Op, All) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {1, 224, 224, 3}, "I");
  Program program("all", {op::all(I)});
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1, I_2, I_3]
) -> (
  _X7
) {
  _X0 = 0;
  _X1 = cmp_eq(I, _X0);
  _X2 = 0;
  _X3 = 1;
  _X4 = cond(_X1, _X2, _X3);
  _X5[] = *(_X4[x0, x1, x2, x3]);
  _X6 = 8;
  _X7 = as_uint(_X5, _X6);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

!int = type tensor<!eltwise.int>
!u8 = type tensor<!eltwise.u8>
module {
  func @all(%arg0: tensor<1x224x224x3x!eltwise.fp32> {tile.name = "I"}) -> !u8 {
    %c1 = "eltwise.sconst"() {value = 1 : i64} : () -> !int
    %c0 = "eltwise.sconst"() {value = 0 : i64} : () -> !int
    %0 = "eltwise.cmp_eq"(%arg0, %c0) {type = !eltwise.fp32} : (tensor<1x224x224x3x!eltwise.fp32>, !int) -> tensor<1x224x224x3x!eltwise.bool>
    %1 = "eltwise.select"(%0, %c0, %c1) {type = !eltwise.fp32} : (tensor<1x224x224x3x!eltwise.bool>, !int, !int) -> !int
    %2 = "tile.domain"() ( {
    ^bb0(%arg1: !eltwise.int, %arg2: !eltwise.int, %arg3: !eltwise.int, %arg4: !eltwise.int):	// no predecessors
      %4 = "tile.src_idx_map"(%1, %arg4, %arg3, %arg2, %arg1) : (!int, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %5 = "tile.sink_idx_map"() : () -> !tile.imap
      %6 = "tile.size_map"() : () -> !tile.smap
      "tile.*(x)"(%6, %4, %5) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3"]} : () -> !int
    %3 = "eltwise.cast"(%2) : (!int) -> !u8
    return %3 : !u8
  }
}
)#"));
#endif
}

TEST(Op, Any) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {1, 224, 224, 3}, "I");
  Program program("any", {op::any(I)});
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1, I_2, I_3]
) -> (
  _X12
) {
  _X0 = 0;
  _X1 = cmp_eq(I, _X0);
  _X2 = 0;
  _X3 = 1;
  _X4 = cond(_X1, _X2, _X3);
  _X5[] = +(_X4[x0, x1, x2, x3]);
  _X6 = 0;
  _X7 = cmp_eq(_X5, _X6);
  _X8 = 0;
  _X9 = 1;
  _X10 = cond(_X7, _X8, _X9);
  _X11 = 8;
  _X12 = as_uint(_X10, _X11);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

!int = type tensor<!eltwise.int>
!bool = type tensor<!eltwise.bool>
!u8 = type tensor<!eltwise.u8>
module {
  func @any(%arg0: tensor<1x224x224x3x!eltwise.fp32> {tile.name = "I"}) -> !u8 {
    %c1 = "eltwise.sconst"() {value = 1 : i64} : () -> !int
    %c0 = "eltwise.sconst"() {value = 0 : i64} : () -> !int
    %0 = "eltwise.cmp_eq"(%arg0, %c0) {type = !eltwise.fp32} : (tensor<1x224x224x3x!eltwise.fp32>, !int) -> tensor<1x224x224x3x!eltwise.bool>
    %1 = "eltwise.select"(%0, %c0, %c1) {type = !eltwise.fp32} : (tensor<1x224x224x3x!eltwise.bool>, !int, !int) -> !int
    %2 = "tile.domain"() ( {
    ^bb0(%arg1: !eltwise.int, %arg2: !eltwise.int, %arg3: !eltwise.int, %arg4: !eltwise.int):	// no predecessors
      %6 = "tile.src_idx_map"(%1, %arg4, %arg3, %arg2, %arg1) : (!int, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %7 = "tile.sink_idx_map"() : () -> !tile.imap
      %8 = "tile.size_map"() : () -> !tile.smap
      "tile.+(x)"(%8, %6, %7) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3"]} : () -> !int
    %3 = "eltwise.cmp_eq"(%2, %c0) {type = !eltwise.fp32} : (!int, !int) -> !bool
    %4 = "eltwise.select"(%3, %c0, %c1) {type = !eltwise.fp32} : (!bool, !int, !int) -> !int
    %5 = "eltwise.cast"(%4) : (!int) -> !u8
    return %5 : !u8
  }
}
)#"));
#endif
}

TEST(Op, Argmax) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {1, 224, 224, 3}, "I");
  Program program("argmax", {op::argmax(I)});
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1, I_2, I_3]
) -> (
  _X7
) {
  _X0[] = >(I[x0, x1, x2, x3]);
  _X1 = 1;
  _X2[x0, x1, x2, x3 : 1, 224, 224, 3] = =(_X1[]);
  _X3 = 0;
  _X4 = index(_X2, _X3);
  _X5[] = >(I[x0, x1, x2, x3] == _X0[] ? _X4[x0, x1, x2, x3]);
  _X6 = 32;
  _X7 = as_uint(_X5, _X6);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

!int = type tensor<!eltwise.int>
!fp32 = type tensor<!eltwise.fp32>
!u32 = type tensor<!eltwise.u32>
module {
  func @argmax(%arg0: tensor<1x224x224x3x!eltwise.fp32> {tile.name = "I"}) -> !u32 {
    %c1 = "eltwise.sconst"() {value = 1 : i64} : () -> !int
    %c3 = "tile.affine_const"() {value = 3 : i64} : () -> !eltwise.int
    %c224 = "tile.affine_const"() {value = 224 : i64} : () -> !eltwise.int
    %c1_0 = "tile.affine_const"() {value = 1 : i64} : () -> !eltwise.int
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !eltwise.int, %arg2: !eltwise.int, %arg3: !eltwise.int, %arg4: !eltwise.int):	// no predecessors
      %5 = "tile.src_idx_map"(%arg0, %arg4, %arg3, %arg2, %arg1) : (tensor<1x224x224x3x!eltwise.fp32>, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %6 = "tile.sink_idx_map"() : () -> !tile.imap
      %7 = "tile.size_map"() : () -> !tile.smap
      "tile.>(x)"(%7, %5, %6) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3"]} : () -> !fp32
    %1 = "tile.domain"() ( {
    ^bb0(%arg1: !eltwise.int, %arg2: !eltwise.int, %arg3: !eltwise.int, %arg4: !eltwise.int):	// no predecessors
      %5 = "tile.src_idx_map"(%c1) : (!int) -> !tile.imap
      %6 = "tile.sink_idx_map"(%arg4, %arg3, %arg2, %arg1) : (!eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %7 = "tile.size_map"(%c1_0, %c224, %c224, %c3) : (!eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.smap
      "tile.=(x)"(%7, %5, %6) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3"]} : () -> tensor<1x224x224x3x!eltwise.int>
    %2 = "tile.index"(%1) {dim = 0 : i64} : (tensor<1x224x224x3x!eltwise.int>) -> tensor<1x224x224x3x!eltwise.int>
    %3 = "tile.domain"() ( {
    ^bb0(%arg1: !eltwise.int, %arg2: !eltwise.int, %arg3: !eltwise.int, %arg4: !eltwise.int):	// no predecessors
      %5 = "tile.src_idx_map"(%arg0, %arg4, %arg3, %arg2, %arg1) : (tensor<1x224x224x3x!eltwise.fp32>, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %6 = "tile.src_idx_map"(%0) : (!fp32) -> !tile.imap
      %7 = "tile.src_idx_map"(%2, %arg4, %arg3, %arg2, %arg1) : (tensor<1x224x224x3x!eltwise.int>, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %8 = "tile.sink_idx_map"() : () -> !tile.imap
      %9 = "tile.size_map"() : () -> !tile.smap
      "tile.>(x==y?z)"(%9, %5, %6, %7, %8) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3"]} : () -> !int
    %4 = "eltwise.cast"(%3) : (!int) -> !u32
    return %4 : !u32
  }
}
)#"));
#endif
}

TEST(Op, BinaryCrossentropy) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, "I");
  auto O = Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, "O");
  Program program("binary_crossentropy", {op::binary_crossentropy(I, O, 0.0)});
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1, I_2, I_3],
  O[O_0, O_1, O_2, O_3]
) -> (
  _X17
) {
  _X0 = ident(I);
  _X1 = 0.000000;
  _X2 = cmp_gt(O, _X1);
  _X3 = cond(_X2, O, _X1);
  _X4 = 1.000000;
  _X5 = cmp_lt(_X3, _X4);
  _X6 = cond(_X5, _X3, _X4);
  _X7 = neg(_X0);
  _X8 = log(_X6);
  _X9 = mul(_X7, _X8);
  _X10 = 1;
  _X11 = sub(_X10, _X0);
  _X12 = 1;
  _X13 = sub(_X12, _X6);
  _X14 = log(_X13);
  _X15 = mul(_X11, _X14);
  _X16 = sub(_X9, _X15);
  _X17 = ident(_X16);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

!float = type tensor<!eltwise.float>
!int = type tensor<!eltwise.int>
module {
  func @binary_crossentropy(%arg0: tensor<7x7x3x64x!eltwise.fp32> {tile.name = "O"}, %arg1: tensor<7x7x3x64x!eltwise.fp32> {tile.name = "I"}) -> tensor<7x7x3x64x!eltwise.fp32> {
    %cst = "eltwise.sconst"() {value = 1.000000e+00 : f64} : () -> !float
    %cst_0 = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !float
    %c1 = "eltwise.sconst"() {value = 1 : i64} : () -> !int
    %0 = "eltwise.cmp_gt"(%arg0, %cst_0) {type = !eltwise.fp32} : (tensor<7x7x3x64x!eltwise.fp32>, !float) -> tensor<7x7x3x64x!eltwise.bool>
    %1 = "eltwise.select"(%0, %arg0, %cst_0) {type = !eltwise.fp32} : (tensor<7x7x3x64x!eltwise.bool>, tensor<7x7x3x64x!eltwise.fp32>, !float) -> tensor<7x7x3x64x!eltwise.fp32>
    %2 = "eltwise.cmp_lt"(%1, %cst) {type = !eltwise.fp32} : (tensor<7x7x3x64x!eltwise.fp32>, !float) -> tensor<7x7x3x64x!eltwise.bool>
    %3 = "eltwise.select"(%2, %1, %cst) {type = !eltwise.fp32} : (tensor<7x7x3x64x!eltwise.bool>, tensor<7x7x3x64x!eltwise.fp32>, !float) -> tensor<7x7x3x64x!eltwise.fp32>
    %4 = "eltwise.sub"(%c1, %3) {type = !eltwise.fp32} : (!int, tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.fp32>
    %5 = "eltwise.log"(%4) {type = !eltwise.fp32} : (tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.fp32>
    %6 = "eltwise.ident"(%arg1) {type = !eltwise.fp32} : (tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.fp32>
    %7 = "eltwise.sub"(%c1, %6) {type = !eltwise.fp32} : (!int, tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.fp32>
    %8 = "eltwise.mul"(%7, %5) {type = !eltwise.fp32} : (tensor<7x7x3x64x!eltwise.fp32>, tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.fp32>
    %9 = "eltwise.log"(%3) {type = !eltwise.fp32} : (tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.fp32>
    %10 = "eltwise.neg"(%6) {type = !eltwise.fp32} : (tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.fp32>
    %11 = "eltwise.mul"(%10, %9) {type = !eltwise.fp32} : (tensor<7x7x3x64x!eltwise.fp32>, tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.fp32>
    %12 = "eltwise.sub"(%11, %8) {type = !eltwise.fp32} : (tensor<7x7x3x64x!eltwise.fp32>, tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.fp32>
    %13 = "eltwise.ident"(%12) {type = !eltwise.fp32} : (tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.fp32>
    return %13 : tensor<7x7x3x64x!eltwise.fp32>
  }
}
)#"));
#endif
}

TEST(Op, Clip) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, "I");
  auto raw_min = Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, "raw_min");
  auto raw_max = Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, "raw_max");
  Program program("clip", {op::clip(I, raw_min, raw_max)});
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1, I_2, I_3],
  raw_min[raw_min_0, raw_min_1, raw_min_2, raw_min_3],
  raw_max[raw_max_0, raw_max_1, raw_max_2, raw_max_3]
) -> (
  _X3
) {
  _X0 = cmp_gt(I, raw_min);
  _X1 = cond(_X0, I, raw_min);
  _X2 = cmp_lt(_X1, raw_max);
  _X3 = cond(_X2, _X1, raw_max);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

module {
  func @clip(%arg0: tensor<7x7x3x64x!eltwise.fp32> {tile.name = "raw_max"}, %arg1: tensor<7x7x3x64x!eltwise.fp32> {tile.name = "raw_min"}, %arg2: tensor<7x7x3x64x!eltwise.fp32> {tile.name = "I"}) -> tensor<7x7x3x64x!eltwise.fp32> {
    %0 = "eltwise.cmp_gt"(%arg2, %arg1) {type = !eltwise.fp32} : (tensor<7x7x3x64x!eltwise.fp32>, tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.bool>
    %1 = "eltwise.select"(%0, %arg2, %arg1) {type = !eltwise.fp32} : (tensor<7x7x3x64x!eltwise.bool>, tensor<7x7x3x64x!eltwise.fp32>, tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.fp32>
    %2 = "eltwise.cmp_lt"(%1, %arg0) {type = !eltwise.fp32} : (tensor<7x7x3x64x!eltwise.fp32>, tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.bool>
    %3 = "eltwise.select"(%2, %1, %arg0) {type = !eltwise.fp32} : (tensor<7x7x3x64x!eltwise.bool>, tensor<7x7x3x64x!eltwise.fp32>, tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.fp32>
    return %3 : tensor<7x7x3x64x!eltwise.fp32>
  }
}
)#"));
#endif
}

TEST(Op, Concatenate) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, "A");
  auto B = Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, "B");
  Program program("concatenate", {op::concatenate({A, B}, 2)});
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1, A_2, A_3],
  B[B_0, B_1, B_2, B_3]
) -> (
  _X2
) {
  _X0[n0, n1, a, n3 : 7, 7, 6, 64] = =(A[n0, n1, a, n3]);
  _X1[n0, n1, 3 + a, n3 : 7, 7, 6, 64] = =(B[n0, n1, a, n3]);
  _X2 = add(_X0, _X1);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

module {
  func @concatenate(%arg0: tensor<7x7x3x64x!eltwise.fp32> {tile.name = "B"}, %arg1: tensor<7x7x3x64x!eltwise.fp32> {tile.name = "A"}) -> tensor<7x7x6x64x!eltwise.fp32> {
    %c6 = "tile.affine_const"() {value = 6 : i64} : () -> !eltwise.int
    %c64 = "tile.affine_const"() {value = 64 : i64} : () -> !eltwise.int
    %c3 = "tile.affine_const"() {value = 3 : i64} : () -> !eltwise.int
    %c7 = "tile.affine_const"() {value = 7 : i64} : () -> !eltwise.int
    %0 = "tile.domain"() ( {
    ^bb0(%arg2: !eltwise.int, %arg3: !eltwise.int, %arg4: !eltwise.int, %arg5: !eltwise.int):	// no predecessors
      %3 = "tile.src_idx_map"(%arg0, %arg5, %arg4, %arg3, %arg2) : (tensor<7x7x3x64x!eltwise.fp32>, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %4 = "tile.affine_add"(%arg3, %c3) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %5 = "tile.sink_idx_map"(%arg5, %arg4, %4, %arg2) : (!eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %6 = "tile.size_map"(%c7, %c7, %c6, %c64) : (!eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.smap
      "tile.=(x)"(%6, %3, %5) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["n3", "a", "n1", "n0"]} : () -> tensor<7x7x6x64x!eltwise.fp32>
    %1 = "tile.domain"() ( {
    ^bb0(%arg2: !eltwise.int, %arg3: !eltwise.int, %arg4: !eltwise.int, %arg5: !eltwise.int):	// no predecessors
      %3 = "tile.src_idx_map"(%arg1, %arg5, %arg4, %arg3, %arg2) : (tensor<7x7x3x64x!eltwise.fp32>, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %4 = "tile.sink_idx_map"(%arg5, %arg4, %arg3, %arg2) : (!eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %5 = "tile.size_map"(%c7, %c7, %c6, %c64) : (!eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.smap
      "tile.=(x)"(%5, %3, %4) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["n3", "a", "n1", "n0"]} : () -> tensor<7x7x6x64x!eltwise.fp32>
    %2 = "eltwise.add"(%1, %0) {type = !eltwise.fp32} : (tensor<7x7x6x64x!eltwise.fp32>, tensor<7x7x6x64x!eltwise.fp32>) -> tensor<7x7x6x64x!eltwise.fp32>
    return %2 : tensor<7x7x6x64x!eltwise.fp32>
  }
}
)#"));
#endif
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
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1, I_2, I_3],
  K[K_0, K_1, K_2, K_3]
) -> (
  conv
) {
  conv[n, x0, x1, co : 1, 112, 112, 64] = +(I[n, -3 + k0 + 2*x0, -3 + k1 + 2*x1, ci] * K[k0, k1, ci, co]);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

module {
  func @convolution(%arg0: tensor<1x224x224x3x!eltwise.fp32> {tile.name = "I"}, %arg1: tensor<7x7x3x64x!eltwise.fp32> {tile.name = "K"}) -> tensor<1x112x112x64x!eltwise.fp32> {
    %c2 = "tile.affine_const"() {value = 2 : i64} : () -> !eltwise.int
    %c3 = "tile.affine_const"() {value = 3 : i64} : () -> !eltwise.int
    %c64 = "tile.affine_const"() {value = 64 : i64} : () -> !eltwise.int
    %c112 = "tile.affine_const"() {value = 112 : i64} : () -> !eltwise.int
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> !eltwise.int
    %0 = "tile.domain"() ( {
    ^bb0(%arg2: !eltwise.int, %arg3: !eltwise.int, %arg4: !eltwise.int, %arg5: !eltwise.int, %arg6: !eltwise.int, %arg7: !eltwise.int, %arg8: !eltwise.int):	// no predecessors
      %1 = "tile.affine_mul"(%arg4, %c2) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %2 = "tile.affine_add"(%1, %arg3) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %3 = "tile.affine_sub"(%2, %c3) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %4 = "tile.affine_mul"(%arg6, %c2) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %5 = "tile.affine_add"(%4, %arg5) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %6 = "tile.affine_sub"(%5, %c3) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %7 = "tile.src_idx_map"(%arg0, %arg7, %6, %3, %arg2) : (tensor<1x224x224x3x!eltwise.fp32>, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %8 = "tile.src_idx_map"(%arg1, %arg5, %arg3, %arg2, %arg8) : (tensor<7x7x3x64x!eltwise.fp32>, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %9 = "tile.sink_idx_map"(%arg7, %arg6, %arg4, %arg8) : (!eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %10 = "tile.size_map"(%c1, %c112, %c112, %c64) : (!eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.smap
      "tile.+(x*y)"(%10, %7, %8, %9) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["ci", "k1", "x1", "k0", "x0", "n", "co"], name = "conv"} : () -> tensor<1x112x112x64x!eltwise.fp32>
    return %0 : tensor<1x112x112x64x!eltwise.fp32>
  }
}
)#"));
#endif
}

TEST(Op, CumProd) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, "I");
  Program program("cumprod", {op::cumprod(I, 2)});
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1, I_2, I_3]
) -> (
  _X0
) {
  _X0[x0, x1, x2, x4 : 7, 7, 3, 64] = *(I[x0, x1, x2 - x3, x4]), x3 < 3;
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

module {
  func @cumprod(%arg0: tensor<7x7x3x64x!eltwise.fp32> {tile.name = "I"}) -> tensor<7x7x3x64x!eltwise.fp32> {
    %c64 = "tile.affine_const"() {value = 64 : i64} : () -> !eltwise.int
    %c7 = "tile.affine_const"() {value = 7 : i64} : () -> !eltwise.int
    %c3 = "tile.affine_const"() {value = 3 : i64} : () -> !eltwise.int
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !eltwise.int, %arg2: !eltwise.int, %arg3: !eltwise.int, %arg4: !eltwise.int, %arg5: !eltwise.int):	// no predecessors
      %1 = "tile.affine_sub"(%arg3, %arg2) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %2 = "tile.src_idx_map"(%arg0, %arg5, %arg4, %1, %arg1) : (tensor<7x7x3x64x!eltwise.fp32>, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %3 = "tile.sink_idx_map"(%arg5, %arg4, %arg3, %arg1) : (!eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %4 = "tile.size_map"(%c7, %c7, %c3, %c64) : (!eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.smap
      "tile.constraint"(%arg2, %c3) ( {
        "tile.*(x)"(%4, %2, %3) : (!tile.smap, !tile.imap, !tile.imap) -> ()
      }) : (!eltwise.int, !eltwise.int) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3", "x4"]} : () -> tensor<7x7x3x64x!eltwise.fp32>
    return %0 : tensor<7x7x3x64x!eltwise.fp32>
  }
}
)#"));
#endif
}

TEST(Op, CumSum) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, "I");
  Program program("cumsum", {op::cumsum(I, 2)});
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1, I_2, I_3]
) -> (
  _X0
) {
  _X0[x0, x1, x2, x4 : 7, 7, 3, 64] = +(I[x0, x1, x2 - x3, x4]), x3 < 3;
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

module {
  func @cumsum(%arg0: tensor<7x7x3x64x!eltwise.fp32> {tile.name = "I"}) -> tensor<7x7x3x64x!eltwise.fp32> {
    %c64 = "tile.affine_const"() {value = 64 : i64} : () -> !eltwise.int
    %c7 = "tile.affine_const"() {value = 7 : i64} : () -> !eltwise.int
    %c3 = "tile.affine_const"() {value = 3 : i64} : () -> !eltwise.int
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !eltwise.int, %arg2: !eltwise.int, %arg3: !eltwise.int, %arg4: !eltwise.int, %arg5: !eltwise.int):	// no predecessors
      %1 = "tile.affine_sub"(%arg3, %arg2) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %2 = "tile.src_idx_map"(%arg0, %arg5, %arg4, %1, %arg1) : (tensor<7x7x3x64x!eltwise.fp32>, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %3 = "tile.sink_idx_map"(%arg5, %arg4, %arg3, %arg1) : (!eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %4 = "tile.size_map"(%c7, %c7, %c3, %c64) : (!eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.smap
      "tile.constraint"(%arg2, %c3) ( {
        "tile.+(x)"(%4, %2, %3) : (!tile.smap, !tile.imap, !tile.imap) -> ()
      }) : (!eltwise.int, !eltwise.int) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3", "x4"]} : () -> tensor<7x7x3x64x!eltwise.fp32>
    return %0 : tensor<7x7x3x64x!eltwise.fp32>
  }
}
)#"));
#endif
}

TEST(Op, Dot) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, "I");
  auto K = Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, "K");
  Program program("dot", {op::dot(I, K)});
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1, I_2, I_3],
  K[K_0, K_1, K_2, K_3]
) -> (
  _X0
) {
  _X0[x0, x1, x2, x4, x5, x6 : 7, 7, 3, 7, 7, 64] = +(I[x0, x1, x2, x3] * K[x4, x5, x3, x6]);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

module {
  func @dot(%arg0: tensor<7x7x3x64x!eltwise.fp32> {tile.name = "I"}, %arg1: tensor<7x7x3x64x!eltwise.fp32> {tile.name = "K"}) -> tensor<7x7x3x7x7x64x!eltwise.fp32> {
    %c64 = "tile.affine_const"() {value = 64 : i64} : () -> !eltwise.int
    %c3 = "tile.affine_const"() {value = 3 : i64} : () -> !eltwise.int
    %c7 = "tile.affine_const"() {value = 7 : i64} : () -> !eltwise.int
    %0 = "tile.domain"() ( {
    ^bb0(%arg2: !eltwise.int, %arg3: !eltwise.int, %arg4: !eltwise.int, %arg5: !eltwise.int, %arg6: !eltwise.int, %arg7: !eltwise.int, %arg8: !eltwise.int):	// no predecessors
      %1 = "tile.src_idx_map"(%arg0, %arg5, %arg4, %arg3, %arg2) : (tensor<7x7x3x64x!eltwise.fp32>, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %2 = "tile.src_idx_map"(%arg1, %arg8, %arg7, %arg2, %arg6) : (tensor<7x7x3x64x!eltwise.fp32>, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %3 = "tile.sink_idx_map"(%arg5, %arg4, %arg3, %arg8, %arg7, %arg6) : (!eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %4 = "tile.size_map"(%c7, %c7, %c3, %c7, %c7, %c64) : (!eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.smap
      "tile.+(x*y)"(%4, %1, %2, %3) : (!tile.smap, !tile.imap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3", "x4", "x5", "x6"]} : () -> tensor<7x7x3x7x7x64x!eltwise.fp32>
    return %0 : tensor<7x7x3x7x7x64x!eltwise.fp32>
  }
}
)#"));
#endif
}

TEST(Op, Elu) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, "I");
  Program program("elu", {op::elu(I, 0.1)});
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1, I_2, I_3]
) -> (
  _X7
) {
  _X0 = 0;
  _X1 = cmp_lt(I, _X0);
  _X2 = 0.100000;
  _X3 = exp(I);
  _X4 = mul(_X2, _X3);
  _X5 = 0.100000;
  _X6 = sub(_X4, _X5);
  _X7 = cond(_X1, _X6, I);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

!float = type tensor<!eltwise.float>
!int = type tensor<!eltwise.int>
module {
  func @elu(%arg0: tensor<7x7x3x64x!eltwise.fp32> {tile.name = "I"}) -> tensor<7x7x3x64x!eltwise.fp32> {
    %cst = "eltwise.sconst"() {value = 0.10000000149011612 : f64} : () -> !float
    %c0 = "eltwise.sconst"() {value = 0 : i64} : () -> !int
    %0 = "eltwise.exp"(%arg0) {type = !eltwise.fp32} : (tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.fp32>
    %1 = "eltwise.mul"(%0, %cst) {type = !eltwise.fp32} : (tensor<7x7x3x64x!eltwise.fp32>, !float) -> tensor<7x7x3x64x!eltwise.fp32>
    %2 = "eltwise.sub"(%1, %cst) {type = !eltwise.fp32} : (tensor<7x7x3x64x!eltwise.fp32>, !float) -> tensor<7x7x3x64x!eltwise.fp32>
    %3 = "eltwise.cmp_lt"(%arg0, %c0) {type = !eltwise.fp32} : (tensor<7x7x3x64x!eltwise.fp32>, !int) -> tensor<7x7x3x64x!eltwise.bool>
    %4 = "eltwise.select"(%3, %2, %arg0) {type = !eltwise.fp32} : (tensor<7x7x3x64x!eltwise.bool>, tensor<7x7x3x64x!eltwise.fp32>, tensor<7x7x3x64x!eltwise.fp32>) -> tensor<7x7x3x64x!eltwise.fp32>
    return %4 : tensor<7x7x3x64x!eltwise.fp32>
  }
}
)#"));
#endif
}

TEST(Op, ExpandDims) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, "I");
  Program program("expand_dims", {op::expand_dims(I, 2)});
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1, I_2, I_3]
) -> (
  _X0
) {
  _X0[n0, n1, a, n2, n3 : 7, 7, 1, 3, 64] = =(I[n0, n1, n2, n3]);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

module {
  func @expand_dims(%arg0: tensor<7x7x3x64x!eltwise.fp32> {tile.name = "I"}) -> tensor<7x7x1x3x64x!eltwise.fp32> {
    %c64 = "tile.affine_const"() {value = 64 : i64} : () -> !eltwise.int
    %c3 = "tile.affine_const"() {value = 3 : i64} : () -> !eltwise.int
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> !eltwise.int
    %c7 = "tile.affine_const"() {value = 7 : i64} : () -> !eltwise.int
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !eltwise.int, %arg2: !eltwise.int, %arg3: !eltwise.int, %arg4: !eltwise.int, %arg5: !eltwise.int):	// no predecessors
      %1 = "tile.src_idx_map"(%arg0, %arg4, %arg3, %arg2, %arg1) : (tensor<7x7x3x64x!eltwise.fp32>, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %2 = "tile.sink_idx_map"(%arg4, %arg3, %arg5, %arg2, %arg1) : (!eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %3 = "tile.size_map"(%c7, %c7, %c1, %c3, %c64) : (!eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.smap
      "tile.=(x)"(%3, %1, %2) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["n3", "n2", "n1", "n0", "a"]} : () -> tensor<7x7x1x3x64x!eltwise.fp32>
    return %0 : tensor<7x7x1x3x64x!eltwise.fp32>
  }
}
)#"));
#endif
}

TEST(Op, Flip) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {7, 7, 3, 64}, "I");
  Program program("flip", {op::flip(I, 2)});
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1, I_2, I_3]
) -> (
  _X0
) {
  _X0[x0, x1, 2 - x2, x3 : 7, 7, 3, 64] = =(I[x0, x1, x2, x3]);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

module {
  func @flip(%arg0: tensor<7x7x3x64x!eltwise.fp32> {tile.name = "I"}) -> tensor<7x7x3x64x!eltwise.fp32> {
    %c2 = "tile.affine_const"() {value = 2 : i64} : () -> !eltwise.int
    %c3 = "tile.affine_const"() {value = 3 : i64} : () -> !eltwise.int
    %c64 = "tile.affine_const"() {value = 64 : i64} : () -> !eltwise.int
    %c7 = "tile.affine_const"() {value = 7 : i64} : () -> !eltwise.int
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !eltwise.int, %arg2: !eltwise.int, %arg3: !eltwise.int, %arg4: !eltwise.int):	// no predecessors
      %1 = "tile.src_idx_map"(%arg0, %arg4, %arg3, %arg2, %arg1) : (tensor<7x7x3x64x!eltwise.fp32>, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %2 = "tile.affine_sub"(%c2, %arg2) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %3 = "tile.sink_idx_map"(%arg4, %arg3, %2, %arg1) : (!eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %4 = "tile.size_map"(%c7, %c7, %c3, %c64) : (!eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.smap
      "tile.=(x)"(%4, %1, %3) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3"]} : () -> tensor<7x7x3x64x!eltwise.fp32>
    return %0 : tensor<7x7x3x64x!eltwise.fp32>
  }
}
)#"));
#endif
}

TEST(Op, HardSigmoid) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  Program program("hard_sigmoid", {op::hard_sigmoid(A, 0.05)});
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1]
) -> (
  _X11
) {
  _X0 = -10.000000;
  _X1 = cmp_lt(A, _X0);
  _X2 = 0.000000;
  _X3 = 10.000000;
  _X4 = cmp_gt(A, _X3);
  _X5 = 1.000000;
  _X6 = 0.050000;
  _X7 = mul(_X6, A);
  _X8 = 0.500000;
  _X9 = add(_X7, _X8);
  _X10 = cond(_X4, _X5, _X9);
  _X11 = cond(_X1, _X2, _X10);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

!float = type tensor<!eltwise.float>
module {
  func @hard_sigmoid(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "A"}) -> tensor<10x20x!eltwise.fp32> {
    %cst = "eltwise.sconst"() {value = 5.000000e-01 : f64} : () -> !float
    %cst_0 = "eltwise.sconst"() {value = 0.05000000074505806 : f64} : () -> !float
    %cst_1 = "eltwise.sconst"() {value = 1.000000e+00 : f64} : () -> !float
    %cst_2 = "eltwise.sconst"() {value = 9.9999998509883898 : f64} : () -> !float
    %cst_3 = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !float
    %cst_4 = "eltwise.sconst"() {value = -9.9999998509883898 : f64} : () -> !float
    %0 = "eltwise.mul"(%arg0, %cst_0) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.fp32>, !float) -> tensor<10x20x!eltwise.fp32>
    %1 = "eltwise.add"(%0, %cst) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.fp32>, !float) -> tensor<10x20x!eltwise.fp32>
    %2 = "eltwise.cmp_gt"(%arg0, %cst_2) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.fp32>, !float) -> tensor<10x20x!eltwise.bool>
    %3 = "eltwise.select"(%2, %cst_1, %1) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.bool>, !float, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    %4 = "eltwise.cmp_lt"(%arg0, %cst_4) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.fp32>, !float) -> tensor<10x20x!eltwise.bool>
    %5 = "eltwise.select"(%4, %cst_3, %3) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.bool>, !float, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    return %5 : tensor<10x20x!eltwise.fp32>
  }
}
)#"));
#endif
}

TEST(Op, ImageResize) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {1, 224, 224, 3}, "I");
  auto image_resize = op::image_resize(I, std::vector<int>{5, 4}, "bilinear", "nxc");
  Program program("image_resize", {image_resize});
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1, I_2, I_3]
) -> (
  _X7
) {
  _X0 = 0.200000;
  _X1[y : 5] = =(_X0[]);
  _X2[y : 9] = +(_X1[-4 + j + y]), j < 5;
  _X3 = 0.250000;
  _X4[x : 4] = =(_X3[]);
  _X5[x : 7] = +(_X4[-3 + i + x]), i < 4;
  _X6[y, x : 9, 7] = =(_X2[y] * _X5[x]);
  _X7[x0, -4 + j + 5*x1, -3 + i + 4*x2, x3 : 1, 1120, 896, 3] = +(I[x0, x1, x2, x3] * _X6[j, i]);
}
)"));
#endif
#ifdef PLAIDML_MLIR
#endif
}

TEST(Op, Max) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {1, 224, 224, 3}, "I");
  Program program("max", {op::max(I)});  // NOLINT(build/include_what_you_use)
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1, I_2, I_3]
) -> (
  _X0
) {
  _X0[] = >(I[x0, x1, x2, x3]);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

!fp32 = type tensor<!eltwise.fp32>
module {
  func @max(%arg0: tensor<1x224x224x3x!eltwise.fp32> {tile.name = "I"}) -> !fp32 {
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !eltwise.int, %arg2: !eltwise.int, %arg3: !eltwise.int, %arg4: !eltwise.int):	// no predecessors
      %1 = "tile.src_idx_map"(%arg0, %arg4, %arg3, %arg2, %arg1) : (tensor<1x224x224x3x!eltwise.fp32>, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %2 = "tile.sink_idx_map"() : () -> !tile.imap
      %3 = "tile.size_map"() : () -> !tile.smap
      "tile.>(x)"(%3, %1, %2) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3"]} : () -> !fp32
    return %0 : !fp32
  }
}
)#"));
#endif
}

TEST(Op, Maximum) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  auto B = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "B");
  Program program("maximum", {op::maximum(A, B)});
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1],
  B[B_0, B_1]
) -> (
  _X1
) {
  _X0 = cmp_lt(A, B);
  _X1 = cond(_X0, B, A);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

module {
  func @maximum(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "A"}, %arg1: tensor<10x20x!eltwise.fp32> {tile.name = "B"}) -> tensor<10x20x!eltwise.fp32> {
    %0 = "eltwise.cmp_lt"(%arg0, %arg1) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.fp32>, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.bool>
    %1 = "eltwise.select"(%0, %arg1, %arg0) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.bool>, tensor<10x20x!eltwise.fp32>, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    return %1 : tensor<10x20x!eltwise.fp32>
  }
}
)#"));
#endif
}

TEST(Op, Mean) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  Program program("mean", {op::mean(A)});
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1]
) -> (
  _X2
) {
  _X0[] = +(A[x0, x1]);
  _X1 = 200;
  _X2 = div(_X0, _X1);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

!int = type tensor<!eltwise.int>
!fp32 = type tensor<!eltwise.fp32>
module {
  func @mean(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "A"}) -> !fp32 {
    %c200 = "eltwise.sconst"() {value = 200 : i64} : () -> !int
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !eltwise.int, %arg2: !eltwise.int):	// no predecessors
      %2 = "tile.src_idx_map"(%arg0, %arg2, %arg1) : (tensor<10x20x!eltwise.fp32>, !eltwise.int, !eltwise.int) -> !tile.imap
      %3 = "tile.sink_idx_map"() : () -> !tile.imap
      %4 = "tile.size_map"() : () -> !tile.smap
      "tile.+(x)"(%4, %2, %3) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1"]} : () -> !fp32
    %1 = "eltwise.div"(%0, %c200) {type = !eltwise.fp32} : (!fp32, !int) -> !fp32
    return %1 : !fp32
  }
}
)#"));
#endif
}

TEST(Op, Min) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  Program program("min", {op::min(A)});  // NOLINT(build/include_what_you_use)
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1]
) -> (
  _X0
) {
  _X0[] = <(A[x0, x1]);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

!fp32 = type tensor<!eltwise.fp32>
module {
  func @min(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "A"}) -> !fp32 {
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !eltwise.int, %arg2: !eltwise.int):	// no predecessors
      %1 = "tile.src_idx_map"(%arg0, %arg2, %arg1) : (tensor<10x20x!eltwise.fp32>, !eltwise.int, !eltwise.int) -> !tile.imap
      %2 = "tile.sink_idx_map"() : () -> !tile.imap
      %3 = "tile.size_map"() : () -> !tile.smap
      "tile.<(x)"(%3, %1, %2) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1"]} : () -> !fp32
    return %0 : !fp32
  }
}
)#"));
#endif
}

TEST(Op, Minimum) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  auto B = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "B");
  Program program("minimum", {op::minimum(A, B)});
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1],
  B[B_0, B_1]
) -> (
  _X1
) {
  _X0 = cmp_lt(A, B);
  _X1 = cond(_X0, A, B);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

module {
  func @minimum(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "B"}, %arg1: tensor<10x20x!eltwise.fp32> {tile.name = "A"}) -> tensor<10x20x!eltwise.fp32> {
    %0 = "eltwise.cmp_lt"(%arg1, %arg0) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.fp32>, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.bool>
    %1 = "eltwise.select"(%0, %arg1, %arg0) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.bool>, tensor<10x20x!eltwise.fp32>, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    return %1 : tensor<10x20x!eltwise.fp32>
  }
}
)#"));
#endif
}

TEST(Op, Pool) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20, 30, 40, 50}, "I");
  Program program("pool", {op::pool(I, "sum", {1, 2, 3}, {1, 2, 3}, "none", {1, 2}, "nwc", true, true)});
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1, I_2, I_3, I_4]
) -> (
  _X0
) {
  _X0[x0, x1, x3, x5, x7 : 10, 22, 17, 14, 50] = +(I[x0, -1 + x1 + x2, -2 + 2*x3 + x4, 3*x5 + x6, x7]), x2 < 1, x4 < 2, x6 < 3;
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

module {
  func @pool(%arg0: tensor<10x20x30x40x50x!eltwise.fp32> {tile.name = "I"}) -> tensor<10x22x17x14x50x!eltwise.fp32> {
    %c50 = "tile.affine_const"() {value = 50 : i64} : () -> !eltwise.int
    %c14 = "tile.affine_const"() {value = 14 : i64} : () -> !eltwise.int
    %c17 = "tile.affine_const"() {value = 17 : i64} : () -> !eltwise.int
    %c22 = "tile.affine_const"() {value = 22 : i64} : () -> !eltwise.int
    %c10 = "tile.affine_const"() {value = 10 : i64} : () -> !eltwise.int
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> !eltwise.int
    %c2 = "tile.affine_const"() {value = 2 : i64} : () -> !eltwise.int
    %c3 = "tile.affine_const"() {value = 3 : i64} : () -> !eltwise.int
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !eltwise.int, %arg2: !eltwise.int, %arg3: !eltwise.int, %arg4: !eltwise.int, %arg5: !eltwise.int, %arg6: !eltwise.int, %arg7: !eltwise.int, %arg8: !eltwise.int):	// no predecessors
      %1 = "tile.affine_mul"(%arg3, %c3) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %2 = "tile.affine_add"(%1, %arg2) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %3 = "tile.affine_mul"(%arg5, %c2) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %4 = "tile.affine_add"(%3, %arg4) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %5 = "tile.affine_sub"(%4, %c2) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %6 = "tile.affine_add"(%arg7, %arg6) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %7 = "tile.affine_sub"(%6, %c1) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %8 = "tile.src_idx_map"(%arg0, %arg8, %7, %5, %2, %arg1) : (tensor<10x20x30x40x50x!eltwise.fp32>, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %9 = "tile.sink_idx_map"(%arg8, %arg7, %arg5, %arg3, %arg1) : (!eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %10 = "tile.size_map"(%c10, %c22, %c17, %c14, %c50) : (!eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.smap
      "tile.constraint"(%arg2, %c3) ( {
        "tile.constraint"(%arg4, %c2) ( {
          "tile.constraint"(%arg6, %c1) ( {
            "tile.+(x)"(%10, %8, %9) : (!tile.smap, !tile.imap, !tile.imap) -> ()
          }) : (!eltwise.int, !eltwise.int) -> ()
        }) : (!eltwise.int, !eltwise.int) -> ()
      }) : (!eltwise.int, !eltwise.int) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"]} : () -> tensor<10x22x17x14x50x!eltwise.fp32>
    return %0 : tensor<10x22x17x14x50x!eltwise.fp32>
  }
}
)#"));
#endif
}

TEST(Op, Prod) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  Program program("prod", {op::prod(A)});
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1]
) -> (
  _X0
) {
  _X0[] = *(A[x0, x1]);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

!fp32 = type tensor<!eltwise.fp32>
module {
  func @prod(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "A"}) -> !fp32 {
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !eltwise.int, %arg2: !eltwise.int):	// no predecessors
      %1 = "tile.src_idx_map"(%arg0, %arg2, %arg1) : (tensor<10x20x!eltwise.fp32>, !eltwise.int, !eltwise.int) -> !tile.imap
      %2 = "tile.sink_idx_map"() : () -> !tile.imap
      %3 = "tile.size_map"() : () -> !tile.smap
      "tile.*(x)"(%3, %1, %2) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1"]} : () -> !fp32
    return %0 : !fp32
  }
}
)#"));
#endif
}

TEST(Op, Relu) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "I");
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  auto M = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "M");
  Program program("relu", {op::relu(I).alpha(A).max_value(M).threshold(0.05)});
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1],
  A[A_0, A_1],
  M[M_0, M_1]
) -> (
  _X7
) {
  _X0 = 0.050000;
  _X1 = cmp_lt(I, _X0);
  _X2 = 0.050000;
  _X3 = sub(I, _X2);
  _X4 = mul(A, _X3);
  _X5 = cond(_X1, _X4, I);
  _X6 = cmp_lt(_X5, M);
  _X7 = cond(_X6, _X5, M);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

!float = type tensor<!eltwise.float>
module {
  func @relu(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "M"}, %arg1: tensor<10x20x!eltwise.fp32> {tile.name = "I"}, %arg2: tensor<10x20x!eltwise.fp32> {tile.name = "A"}) -> tensor<10x20x!eltwise.fp32> {
    %cst = "eltwise.sconst"() {value = 0.05000000074505806 : f64} : () -> !float
    %0 = "eltwise.sub"(%arg1, %cst) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.fp32>, !float) -> tensor<10x20x!eltwise.fp32>
    %1 = "eltwise.mul"(%arg2, %0) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.fp32>, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    %2 = "eltwise.cmp_lt"(%arg1, %cst) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.fp32>, !float) -> tensor<10x20x!eltwise.bool>
    %3 = "eltwise.select"(%2, %1, %arg1) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.bool>, tensor<10x20x!eltwise.fp32>, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    %4 = "eltwise.cmp_lt"(%3, %arg0) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.fp32>, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.bool>
    %5 = "eltwise.select"(%4, %3, %arg0) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.bool>, tensor<10x20x!eltwise.fp32>, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    return %5 : tensor<10x20x!eltwise.fp32>
  }
}
)#"));
#endif
}

TEST(Op, ReluNoAlpha) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "I");
  auto M = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "M");
  Program program("relu", {op::relu(I).max_value(M).threshold(0.05)});
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1],
  M[M_0, M_1]
) -> (
  _X8
) {
  _X0 = 0.050000;
  _X1 = cmp_lt(I, _X0);
  _X2 = 0.000000;
  _X3 = 0.050000;
  _X4 = sub(I, _X3);
  _X5 = mul(_X2, _X4);
  _X6 = cond(_X1, _X5, I);
  _X7 = cmp_lt(_X6, M);
  _X8 = cond(_X7, _X6, M);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

!float = type tensor<!eltwise.float>
module {
  func @relu(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "M"}, %arg1: tensor<10x20x!eltwise.fp32> {tile.name = "I"}) -> tensor<10x20x!eltwise.fp32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !float
    %cst_0 = "eltwise.sconst"() {value = 0.05000000074505806 : f64} : () -> !float
    %0 = "eltwise.cmp_lt"(%arg1, %cst_0) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.fp32>, !float) -> tensor<10x20x!eltwise.bool>
    %1 = "eltwise.select"(%0, %cst, %arg1) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.bool>, !float, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    %2 = "eltwise.cmp_lt"(%1, %arg0) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.fp32>, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.bool>
    %3 = "eltwise.select"(%2, %1, %arg0) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.bool>, tensor<10x20x!eltwise.fp32>, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    return %3 : tensor<10x20x!eltwise.fp32>
  }
}
)#"));
#endif
}

TEST(Op, ReluNoMaxValue) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "I");
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  Program program("relu", {op::relu(I).alpha(A).threshold(0.05)});
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1],
  A[A_0, A_1]
) -> (
  _X5
) {
  _X0 = 0.050000;
  _X1 = cmp_lt(I, _X0);
  _X2 = 0.050000;
  _X3 = sub(I, _X2);
  _X4 = mul(A, _X3);
  _X5 = cond(_X1, _X4, I);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

!float = type tensor<!eltwise.float>
module {
  func @relu(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "I"}, %arg1: tensor<10x20x!eltwise.fp32> {tile.name = "A"}) -> tensor<10x20x!eltwise.fp32> {
    %cst = "eltwise.sconst"() {value = 0.05000000074505806 : f64} : () -> !float
    %0 = "eltwise.sub"(%arg0, %cst) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.fp32>, !float) -> tensor<10x20x!eltwise.fp32>
    %1 = "eltwise.mul"(%arg1, %0) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.fp32>, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    %2 = "eltwise.cmp_lt"(%arg0, %cst) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.fp32>, !float) -> tensor<10x20x!eltwise.bool>
    %3 = "eltwise.select"(%2, %1, %arg0) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.bool>, tensor<10x20x!eltwise.fp32>, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    return %3 : tensor<10x20x!eltwise.fp32>
  }
}
)#"));
#endif
}

TEST(Op, ReluOnlyThreshold) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "I");
  Program program("relu", {op::relu(I).threshold(0.05)});
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1]
) -> (
  _X6
) {
  _X0 = 0.050000;
  _X1 = cmp_lt(I, _X0);
  _X2 = 0.000000;
  _X3 = 0.050000;
  _X4 = sub(I, _X3);
  _X5 = mul(_X2, _X4);
  _X6 = cond(_X1, _X5, I);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

!float = type tensor<!eltwise.float>
module {
  func @relu(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "I"}) -> tensor<10x20x!eltwise.fp32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !float
    %cst_0 = "eltwise.sconst"() {value = 0.05000000074505806 : f64} : () -> !float
    %0 = "eltwise.cmp_lt"(%arg0, %cst_0) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.fp32>, !float) -> tensor<10x20x!eltwise.bool>
    %1 = "eltwise.select"(%0, %cst, %arg0) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.bool>, !float, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    return %1 : tensor<10x20x!eltwise.fp32>
  }
}
)#"));
#endif
}

TEST(Op, ReluNoParams) {
  auto I = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "I");
  Program program("relu", {op::relu(I)});
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  I[I_0, I_1]
) -> (
  _X4
) {
  _X0 = 0.000000;
  _X1 = cmp_lt(I, _X0);
  _X2 = 0.000000;
  _X3 = mul(_X2, I);
  _X4 = cond(_X1, _X3, I);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

!float = type tensor<!eltwise.float>
module {
  func @relu(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "I"}) -> tensor<10x20x!eltwise.fp32> {
    %cst = "eltwise.sconst"() {value = 0.000000e+00 : f64} : () -> !float
    %0 = "eltwise.cmp_lt"(%arg0, %cst) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.fp32>, !float) -> tensor<10x20x!eltwise.bool>
    %1 = "eltwise.select"(%0, %cst, %arg0) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.bool>, !float, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    return %1 : tensor<10x20x!eltwise.fp32>
  }
}
)#"));
#endif
}

TEST(Op, Repeat) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {32, 1, 4, 1}, "A");
  auto X = op::repeat(  //
      A,                // tensor to repeat
      3,                // number of repeats
      2);               // axis to repeat
  Program program("repeat", {X});
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1, A_2, A_3]
) -> (
  _X0
) {
  _X0[x0, x1, 3*x2 + x4, x3 : 32, 1, 12, 1] = =(A[x0, x1, x2, x3]), x4 < 3;
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

module {
  func @repeat(%arg0: tensor<32x1x4x1x!eltwise.fp32> {tile.name = "A"}) -> tensor<32x1x12x1x!eltwise.fp32> {
    %c12 = "tile.affine_const"() {value = 12 : i64} : () -> !eltwise.int
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> !eltwise.int
    %c32 = "tile.affine_const"() {value = 32 : i64} : () -> !eltwise.int
    %c3 = "tile.affine_const"() {value = 3 : i64} : () -> !eltwise.int
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !eltwise.int, %arg2: !eltwise.int, %arg3: !eltwise.int, %arg4: !eltwise.int, %arg5: !eltwise.int):	// no predecessors
      %1 = "tile.src_idx_map"(%arg0, %arg4, %arg3, %arg2, %arg1) : (tensor<32x1x4x1x!eltwise.fp32>, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %2 = "tile.affine_mul"(%arg2, %c3) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %3 = "tile.affine_add"(%2, %arg5) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %4 = "tile.sink_idx_map"(%arg4, %arg3, %3, %arg1) : (!eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %5 = "tile.size_map"(%c32, %c1, %c12, %c1) : (!eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.smap
      "tile.constraint"(%arg5, %c3) ( {
        "tile.=(x)"(%5, %1, %4) : (!tile.smap, !tile.imap, !tile.imap) -> ()
      }) : (!eltwise.int, !eltwise.int) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3", "x4"]} : () -> tensor<32x1x12x1x!eltwise.fp32>
    return %0 : tensor<32x1x12x1x!eltwise.fp32>
  }
}
)#"));
#endif
}

TEST(Op, Reshape) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  TensorDim I, J;
  A.bind_dims(I, J);
  Program program("reshape", {op::reshape(A, make_tuple(J, I))});
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1]
) -> (
  _X2
) {
  _X0 = 20;
  _X1 = 10;
  _X2 = reshape(A, _X0, _X1);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

module {
  func @reshape(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "A"}) -> tensor<20x10x!eltwise.fp32> {
    %c10 = "tile.affine_const"() {value = 10 : i64} : () -> !eltwise.int
    %c20 = "tile.affine_const"() {value = 20 : i64} : () -> !eltwise.int
    %0 = "tile.reshape"(%arg0, %c20, %c10) : (tensor<10x20x!eltwise.fp32>, !eltwise.int, !eltwise.int) -> tensor<20x10x!eltwise.fp32>
    return %0 : tensor<20x10x!eltwise.fp32>
  }
}
)#"));
#endif
}

TEST(Op, Sigmoid) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10}, "A");
  Program program("sigmoid", {op::sigmoid(A)});
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0]
) -> (
  _X7
) {
  _X0 = ident(A);
  _X1 = 1.000000;
  _X2 = 1.000000;
  _X3 = neg(_X0);
  _X4 = exp(_X3);
  _X5 = add(_X2, _X4);
  _X6 = div(_X1, _X5);
  _X7 = ident(_X6);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

!float = type tensor<!eltwise.float>
module {
  func @sigmoid(%arg0: tensor<10x!eltwise.fp32> {tile.name = "A"}) -> tensor<10x!eltwise.fp32> {
    %cst = "eltwise.sconst"() {value = 1.000000e+00 : f64} : () -> !float
    %0 = "eltwise.ident"(%arg0) {type = !eltwise.fp32} : (tensor<10x!eltwise.fp32>) -> tensor<10x!eltwise.fp32>
    %1 = "eltwise.neg"(%0) {type = !eltwise.fp32} : (tensor<10x!eltwise.fp32>) -> tensor<10x!eltwise.fp32>
    %2 = "eltwise.exp"(%1) {type = !eltwise.fp32} : (tensor<10x!eltwise.fp32>) -> tensor<10x!eltwise.fp32>
    %3 = "eltwise.add"(%2, %cst) {type = !eltwise.fp32} : (tensor<10x!eltwise.fp32>, !float) -> tensor<10x!eltwise.fp32>
    %4 = "eltwise.div"(%cst, %3) {type = !eltwise.fp32} : (!float, tensor<10x!eltwise.fp32>) -> tensor<10x!eltwise.fp32>
    %5 = "eltwise.ident"(%4) {type = !eltwise.fp32} : (tensor<10x!eltwise.fp32>) -> tensor<10x!eltwise.fp32>
    return %5 : tensor<10x!eltwise.fp32>
  }
}
)#"));
#endif
}

TEST(Op, Slice) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  auto X = op::slice(  //
      A,               // tensor to perform spatial padding on
      {2, 10});        // slices
  Program program("slice", {X});
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1]
) -> (
  _X0
) {
  _X0[] = =(A[2, 10]);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

!fp32 = type tensor<!eltwise.fp32>
module {
  func @slice(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "A"}) -> !fp32 {
    %c10 = "tile.affine_const"() {value = 10 : i64} : () -> !eltwise.int
    %c2 = "tile.affine_const"() {value = 2 : i64} : () -> !eltwise.int
    %0 = "tile.domain"() ( {
      %1 = "tile.src_idx_map"(%arg0, %c2, %c10) : (tensor<10x20x!eltwise.fp32>, !eltwise.int, !eltwise.int) -> !tile.imap
      %2 = "tile.sink_idx_map"() : () -> !tile.imap
      %3 = "tile.size_map"() : () -> !tile.smap
      "tile.=(x)"(%3, %1, %2) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = []} : () -> !fp32
    return %0 : !fp32
  }
}
)#"));
#endif
}

TEST(Op, Softmax) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  Program program("softmax", {op::softmax(A, 1)});
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1]
) -> (
  _X6
) {
  _X0 = ident(A);
  _X1[x0, 0 : 10, 1] = >(_X0[x0, x1]);
  _X2 = sub(_X0, _X1);
  _X3 = exp(_X2);
  _X4[x0, 0 : 10, 1] = +(_X3[x0, x1]);
  _X5 = div(_X3, _X4);
  _X6 = ident(_X5);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

module {
  func @softmax(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "A"}) -> tensor<10x20x!eltwise.fp32> {
    %c0 = "tile.affine_const"() {value = 0 : i64} : () -> !eltwise.int
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> !eltwise.int
    %c10 = "tile.affine_const"() {value = 10 : i64} : () -> !eltwise.int
    %0 = "eltwise.ident"(%arg0) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    %1 = "tile.domain"() ( {
    ^bb0(%arg1: !eltwise.int, %arg2: !eltwise.int):	// no predecessors
      %7 = "tile.src_idx_map"(%0, %arg2, %arg1) : (tensor<10x20x!eltwise.fp32>, !eltwise.int, !eltwise.int) -> !tile.imap
      %8 = "tile.sink_idx_map"(%arg2, %c0) : (!eltwise.int, !eltwise.int) -> !tile.imap
      %9 = "tile.size_map"(%c10, %c1) : (!eltwise.int, !eltwise.int) -> !tile.smap
      "tile.>(x)"(%9, %7, %8) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1"]} : () -> tensor<10x1x!eltwise.fp32>
    %2 = "eltwise.sub"(%0, %1) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.fp32>, tensor<10x1x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    %3 = "eltwise.exp"(%2) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    %4 = "tile.domain"() ( {
    ^bb0(%arg1: !eltwise.int, %arg2: !eltwise.int):	// no predecessors
      %7 = "tile.src_idx_map"(%3, %arg2, %arg1) : (tensor<10x20x!eltwise.fp32>, !eltwise.int, !eltwise.int) -> !tile.imap
      %8 = "tile.sink_idx_map"(%arg2, %c0) : (!eltwise.int, !eltwise.int) -> !tile.imap
      %9 = "tile.size_map"(%c10, %c1) : (!eltwise.int, !eltwise.int) -> !tile.smap
      "tile.+(x)"(%9, %7, %8) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1"]} : () -> tensor<10x1x!eltwise.fp32>
    %5 = "eltwise.div"(%3, %4) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.fp32>, tensor<10x1x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    %6 = "eltwise.ident"(%5) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    return %6 : tensor<10x20x!eltwise.fp32>
  }
}
)#"));
#endif
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
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1, A_2, A_3]
) -> (
  _X0
) {
  _X0[n, c, 1 + x0, 3 + x1 : 64, 4, 36, 38] = =(A[n, c, x0, x1]);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

module {
  func @spatial_padding(%arg0: tensor<64x4x32x32x!eltwise.fp32> {tile.name = "A"}) -> tensor<64x4x36x38x!eltwise.fp32> {
    %c3 = "tile.affine_const"() {value = 3 : i64} : () -> !eltwise.int
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> !eltwise.int
    %c38 = "tile.affine_const"() {value = 38 : i64} : () -> !eltwise.int
    %c36 = "tile.affine_const"() {value = 36 : i64} : () -> !eltwise.int
    %c4 = "tile.affine_const"() {value = 4 : i64} : () -> !eltwise.int
    %c64 = "tile.affine_const"() {value = 64 : i64} : () -> !eltwise.int
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !eltwise.int, %arg2: !eltwise.int, %arg3: !eltwise.int, %arg4: !eltwise.int):	// no predecessors
      %1 = "tile.src_idx_map"(%arg0, %arg4, %arg3, %arg2, %arg1) : (tensor<64x4x32x32x!eltwise.fp32>, !eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %2 = "tile.affine_add"(%arg1, %c3) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %3 = "tile.affine_add"(%arg2, %c1) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %4 = "tile.sink_idx_map"(%arg4, %arg3, %3, %2) : (!eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.imap
      %5 = "tile.size_map"(%c64, %c4, %c36, %c38) : (!eltwise.int, !eltwise.int, !eltwise.int, !eltwise.int) -> !tile.smap
      "tile.=(x)"(%5, %1, %4) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x1", "x0", "c", "n"]} : () -> tensor<64x4x36x38x!eltwise.fp32>
    return %0 : tensor<64x4x36x38x!eltwise.fp32>
  }
}
)#"));
#endif
}

TEST(Op, Square) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10}, "A");
  Program program("square", {op::square(A)});
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0]
) -> (
  _X0
) {
  _X0 = mul(A, A);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

module {
  func @square(%arg0: tensor<10x!eltwise.fp32> {tile.name = "A"}) -> tensor<10x!eltwise.fp32> {
    %0 = "eltwise.mul"(%arg0, %arg0) {type = !eltwise.fp32} : (tensor<10x!eltwise.fp32>, tensor<10x!eltwise.fp32>) -> tensor<10x!eltwise.fp32>
    return %0 : tensor<10x!eltwise.fp32>
  }
}
)#"));
#endif
}

TEST(Op, Sum) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  Program program("sum", {op::sum(A)});
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1]
) -> (
  _X0
) {
  _X0[] = +(A[x0, x1]);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

!fp32 = type tensor<!eltwise.fp32>
module {
  func @sum(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "A"}) -> !fp32 {
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !eltwise.int, %arg2: !eltwise.int):	// no predecessors
      %1 = "tile.src_idx_map"(%arg0, %arg2, %arg1) : (tensor<10x20x!eltwise.fp32>, !eltwise.int, !eltwise.int) -> !tile.imap
      %2 = "tile.sink_idx_map"() : () -> !tile.imap
      %3 = "tile.size_map"() : () -> !tile.smap
      "tile.+(x)"(%3, %1, %2) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1"]} : () -> !fp32
    return %0 : !fp32
  }
}
)#"));
#endif
}

TEST(Op, Squeeze) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {32, 1, 4, 1}, "A");
  auto X = op::squeeze(  //
      A,                 // tensor to squeeze
      {1, 3});           // axes to squeeze
  Program program("squeeze", {X});
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1, A_2, A_3]
) -> (
  _X2
) {
  _X0 = 32;
  _X1 = 4;
  _X2 = reshape(A, _X0, _X1);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

module {
  func @squeeze(%arg0: tensor<32x1x4x1x!eltwise.fp32> {tile.name = "A"}) -> tensor<32x4x!eltwise.fp32> {
    %c4 = "tile.affine_const"() {value = 4 : i64} : () -> !eltwise.int
    %c32 = "tile.affine_const"() {value = 32 : i64} : () -> !eltwise.int
    %0 = "tile.reshape"(%arg0, %c32, %c4) : (tensor<32x1x4x1x!eltwise.fp32>, !eltwise.int, !eltwise.int) -> tensor<32x4x!eltwise.fp32>
    return %0 : tensor<32x4x!eltwise.fp32>
  }
}
)#"));
#endif
}

TEST(Op, Tile) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  auto X = op::tile(  //
      A,              // tensor to tile
      {5, 4});        // tiling factors
  Program program("tile", {X});
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1]
) -> (
  _X0
) {
  _X0[x0 + 10*x2, x1 + 20*x3 : 50, 80] = =(A[x0, x1]) no_defract;
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

module {
  func @tile(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "A"}) -> tensor<50x80x!eltwise.fp32> {
    %c80 = "tile.affine_const"() {value = 80 : i64} : () -> !eltwise.int
    %c20 = "tile.affine_const"() {value = 20 : i64} : () -> !eltwise.int
    %c50 = "tile.affine_const"() {value = 50 : i64} : () -> !eltwise.int
    %c10 = "tile.affine_const"() {value = 10 : i64} : () -> !eltwise.int
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !eltwise.int, %arg2: !eltwise.int, %arg3: !eltwise.int, %arg4: !eltwise.int):	// no predecessors
      %1 = "tile.src_idx_map"(%arg0, %arg2, %arg1) : (tensor<10x20x!eltwise.fp32>, !eltwise.int, !eltwise.int) -> !tile.imap
      %2 = "tile.affine_mul"(%arg3, %c20) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %3 = "tile.affine_add"(%2, %arg1) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %4 = "tile.affine_mul"(%arg4, %c10) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %5 = "tile.affine_add"(%4, %arg2) : (!eltwise.int, !eltwise.int) -> !eltwise.int
      %6 = "tile.sink_idx_map"(%5, %3) : (!eltwise.int, !eltwise.int) -> !tile.imap
      %7 = "tile.size_map"(%c50, %c80) : (!eltwise.int, !eltwise.int) -> !tile.smap
      "tile.=(x)"(%7, %1, %6) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3"], no_reduce = true} : () -> tensor<50x80x!eltwise.fp32>
    return %0 : tensor<50x80x!eltwise.fp32>
  }
}
)#"));
#endif
}

TEST(Op, Transpose) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  Program program("transpose", {op::transpose(A)});
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1]
) -> (
  _X0
) {
  _X0[x1, x0 : 20, 10] = =(A[x0, x1]);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

module {
  func @transpose(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "A"}) -> tensor<20x10x!eltwise.fp32> {
    %c10 = "tile.affine_const"() {value = 10 : i64} : () -> !eltwise.int
    %c20 = "tile.affine_const"() {value = 20 : i64} : () -> !eltwise.int
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !eltwise.int, %arg2: !eltwise.int):	// no predecessors
      %1 = "tile.src_idx_map"(%arg0, %arg2, %arg1) : (tensor<10x20x!eltwise.fp32>, !eltwise.int, !eltwise.int) -> !tile.imap
      %2 = "tile.sink_idx_map"(%arg1, %arg2) : (!eltwise.int, !eltwise.int) -> !tile.imap
      %3 = "tile.size_map"(%c20, %c10) : (!eltwise.int, !eltwise.int) -> !tile.smap
      "tile.=(x)"(%3, %1, %2) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1"]} : () -> tensor<20x10x!eltwise.fp32>
    return %0 : tensor<20x10x!eltwise.fp32>
  }
}
)#"));
#endif
}

TEST(Op, Variance) {
  auto A = Placeholder(PLAIDML_DATA_FLOAT32, {10, 20}, "A");
  Program program("variance", {op::variance(A)});
  IVLOG(1, program);
#ifdef PLAIDML_AST
  EXPECT_THAT(program, Eq(R"(function (
  A[A_0, A_1]
) -> (
  _X8
) {
  _X0[x2, x3 : 1, 1] = +(A[x0, x1]);
  _X1 = 200;
  _X2 = div(_X0, _X1);
  _X3 = sub(A, _X2);
  _X4 = sub(A, _X2);
  _X5 = mul(_X3, _X4);
  _X6[] = +(_X5[x0, x1]);
  _X7 = 200;
  _X8 = div(_X6, _X7);
}
)"));
#endif
#ifdef PLAIDML_MLIR
  EXPECT_THAT(program, Eq(R"#(

!int = type tensor<!eltwise.int>
!fp32 = type tensor<!eltwise.fp32>
module {
  func @variance(%arg0: tensor<10x20x!eltwise.fp32> {tile.name = "A"}) -> !fp32 {
    %c200 = "eltwise.sconst"() {value = 200 : i64} : () -> !int
    %c1 = "tile.affine_const"() {value = 1 : i64} : () -> !eltwise.int
    %0 = "tile.domain"() ( {
    ^bb0(%arg1: !eltwise.int, %arg2: !eltwise.int, %arg3: !eltwise.int, %arg4: !eltwise.int):	// no predecessors
      %6 = "tile.src_idx_map"(%arg0, %arg2, %arg1) : (tensor<10x20x!eltwise.fp32>, !eltwise.int, !eltwise.int) -> !tile.imap
      %7 = "tile.sink_idx_map"(%arg4, %arg3) : (!eltwise.int, !eltwise.int) -> !tile.imap
      %8 = "tile.size_map"(%c1, %c1) : (!eltwise.int, !eltwise.int) -> !tile.smap
      "tile.+(x)"(%8, %6, %7) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1", "x2", "x3"]} : () -> tensor<1x1x!eltwise.fp32>
    %1 = "eltwise.div"(%0, %c200) {type = !eltwise.fp32} : (tensor<1x1x!eltwise.fp32>, !int) -> tensor<1x1x!eltwise.fp32>
    %2 = "eltwise.sub"(%arg0, %1) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.fp32>, tensor<1x1x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    %3 = "eltwise.mul"(%2, %2) {type = !eltwise.fp32} : (tensor<10x20x!eltwise.fp32>, tensor<10x20x!eltwise.fp32>) -> tensor<10x20x!eltwise.fp32>
    %4 = "tile.domain"() ( {
    ^bb0(%arg1: !eltwise.int, %arg2: !eltwise.int):	// no predecessors
      %6 = "tile.src_idx_map"(%3, %arg2, %arg1) : (tensor<10x20x!eltwise.fp32>, !eltwise.int, !eltwise.int) -> !tile.imap
      %7 = "tile.sink_idx_map"() : () -> !tile.imap
      %8 = "tile.size_map"() : () -> !tile.smap
      "tile.+(x)"(%8, %6, %7) : (!tile.smap, !tile.imap, !tile.imap) -> ()
    }) {idx_names = ["x0", "x1"]} : () -> !fp32
    %5 = "eltwise.div"(%4, %c200) {type = !eltwise.fp32} : (!fp32, !int) -> !fp32
    return %5 : !fp32
  }
}
)#"));
#endif
}

}  // namespace
}  // namespace plaidml::op
