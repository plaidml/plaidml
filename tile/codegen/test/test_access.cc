// Copyright 2018, Intel Corp.

#include <gmock/gmock.h>

#include "tile/codegen/access.h"
#include "tile/codegen/cache.h"
#include "tile/codegen/tile.h"
#include "tile/codegen/vm.h"
#include "tile/lang/compose.h"
#include "tile/lang/gen_stripe.h"
#include "tile/lang/stripe.h"

using ::testing::Eq;

namespace vertexai {
namespace tile {
namespace codegen {
namespace test {

static lang::RunInfo LoadMatMul() {
  const size_t DIM = 5;
  lang::RunInfo runinfo;
  runinfo.program_name = "MatMul";
  runinfo.code = "function (A[M, K], B[K, N]) -> (C) { C[m, n : M, N] = +(A[m, k] * B[k, n]); }";
  runinfo.input_shapes.emplace("A", lang::SimpleShape(lang::DataType::FLOAT32, {DIM, DIM}));
  runinfo.input_shapes.emplace("B", lang::SimpleShape(lang::DataType::FLOAT32, {DIM, DIM}));
  runinfo.output_shapes.emplace("C", lang::SimpleShape(lang::DataType::FLOAT32, {DIM, DIM}));
  return runinfo;
}

TEST(Codegen, Access) {
  auto runinfo = LoadMatMul();
  auto program = GenerateStripe(runinfo);

  auto main = program.mutable_stmts(0)->mutable_block();
  auto kernel = main->mutable_stmts(0)->mutable_block();
  ApplyTile(kernel, {2, 2, 2});
  auto inner = kernel->mutable_stmts(0)->mutable_block();

  auto access = ComputeAccess(*kernel, "A");
  EXPECT_THAT(access.size(), Eq(1));
  AccessPattern expected1 = {false,
                             true,
                             {
                                 {"k", 3, 0},  //
                                 {"m", 3, 0},  //
                                 {"n", 3, 0},  //
                                 {"k", 2, 2},  //
                                 {"m", 2, 2},  //
                                 {"n", 2, 2},  //
                             },
                             {0, {2, 10, 0, 1, 5, 0}},
                             {
                                 {{2, 0, 0, 1, 0, 0}, 5},
                                 {{0, 2, 0, 0, 1, 0}, 5},
                                 {{0, 0, 2, 0, 0, 1}, 5},
                             }};
  EXPECT_THAT(access[0], Eq(expected1));

  access = ComputeAccess(*inner, "A");
  EXPECT_THAT(access.size(), Eq(1));
  AccessPattern expected2 = {false,
                             false,
                             {
                                 {"k", 2, 2},  //
                                 {"m", 2, 2},  //
                                 {"n", 2, 2},  //
                             },
                             {0, {1, 5, 0}},
                             {}};
  EXPECT_THAT(access[0], Eq(expected2));
}

TEST(Codegen, Cache) {
  auto runinfo = LoadMatMul();
  auto program = GenerateStripe(runinfo);

  auto main = program.mutable_stmts(0)->mutable_block();
  auto kernel = main->mutable_stmts(0)->mutable_block();
  ApplyTile(kernel, {2, 2, 2});
  auto inner = kernel->mutable_stmts(0)->mutable_block();
  ApplyCache(inner, "A");
  std::cout << *inner;
  std::cout << "W00t\n";
}

}  // namespace test
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
