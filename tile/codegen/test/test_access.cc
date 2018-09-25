// Copyright 2018, Intel Corp.

#include <gmock/gmock.h>

#include "tile/codegen/access.h"
#include "tile/codegen/cache.h"
#include "tile/codegen/tile.h"
#include "tile/codegen/vm.h"
#include "tile/lang/compose.h"
#include "tile/lang/gen_stripe.h"
#include "tile/stripe/stripe.h"

using ::testing::Eq;

namespace vertexai {
namespace tile {
namespace codegen {
namespace test {

lang::RunInfo LoadMatMul() {
  const size_t DIM = 5;
  lang::RunInfo runinfo;
  runinfo.program_name = "MatMul";
  runinfo.code = "function (A[M, K], B[K, N]) -> (C) { C[m, n : M, N] = +(A[m, k] * B[k, n]); }";
  runinfo.input_shapes.emplace("A", SimpleShape(DataType::FLOAT32, {DIM, DIM}));
  runinfo.input_shapes.emplace("B", SimpleShape(DataType::FLOAT32, {DIM, DIM}));
  runinfo.output_shapes.emplace("C", SimpleShape(DataType::FLOAT32, {DIM, DIM}));
  return runinfo;
}

TEST(Codegen, Access) {
  auto runinfo = LoadMatMul();
  auto program = GenerateStripe(runinfo);

  auto main = std::dynamic_pointer_cast<stripe::Block>(program.stmts[0]);
  auto kernel = std::dynamic_pointer_cast<stripe::Block>(main->stmts[0]);
  ApplyTile(kernel.get(), {2, 2, 2});
  auto inner = std::dynamic_pointer_cast<stripe::Block>(kernel->stmts[0]);

  auto access = ComputeAccess(*kernel, "A");
  ASSERT_THAT(access.size(), Eq(1));
  AccessPattern expected1 = {
      false,  // is_write
      true,   // is_exact
      {
          // idxs
          {"k", 3, 0},  //
          {"m", 3, 0},  //
          {"n", 3, 0},  //
          {"k", 2, 2},  //
          {"m", 2, 2},  //
          {"n", 2, 2},  //
      },
      {
          // access
          0,                   // offset
          {2, 10, 0, 1, 5, 0}  // strides
      },
      {
          // constraints
          {{2, 0, 0, 1, 0, 0}, 5},
          {{0, 2, 0, 0, 1, 0}, 5},
          {{0, 0, 2, 0, 0, 1}, 5},
      }  //
  };
  EXPECT_THAT(access[0], Eq(expected1));

  access = ComputeAccess(*inner, "A");
  EXPECT_THAT(access.size(), Eq(1));
  AccessPattern expected2 = {
      false,  // is_write
      false,  // is_exact
      {
          // idxs
          {"k", 2, 2},  //
          {"m", 2, 2},  //
          {"n", 2, 2},  //
      },
      {0, {1, 5, 0}},  // access
      {}               // constraints
  };
  EXPECT_THAT(access[0], Eq(expected2));
}

TEST(Codegen, CacheInfo) {
  auto c1 = ComputeCacheInfo({{"I", 10, 0}, {"K", 10, 0}, {"Q", 10, 0}, {"J", 3, 0}}, {0, {1, 100, 0, 1}});
  CacheInfo expected1 = {
      {{"I", 10, 0}, {"K", 10, 0}, {"Q", 10, 0}, {"J", 3, 0}},  // idxs
      {0, {1, 100, 0, 1}},                                      // far
      {0, {1, 12, 0, 1}},                                       // near
      {{"I_J", 12, 0}, {"K", 10, 0}},                           // xfer_idxs
      {0, {1, 100}},                                            // xfer_far
      {0, {1, 12}},                                             // xfer_near
  };
  EXPECT_THAT(c1, Eq(expected1));

  auto c2 = ComputeCacheInfo({{"I", 10, 0}, {"K", 10, 0}, {"J", 3, 0}}, {0, {1, 12, 1}});
  CacheInfo expected2 = {
      {{"I", 10, 0}, {"K", 10, 0}, {"J", 3, 0}},  // idxs
      {0, {1, 12, 1}},                            // far
      {0, {1, 12, 1}},                            // near
      {{"I_J_K", 120, 0}},                        // xfer_idxs
      {0, {1}},                                   // xfer_far
      {0, {1}},                                   // xfer_near
  };
  EXPECT_THAT(c2, Eq(expected2));
}

TEST(Codegen, Cache) {
  auto runinfo = LoadMatMul();
  auto program = GenerateStripe(runinfo);

  auto main = std::dynamic_pointer_cast<stripe::Block>(program.stmts[0]);
  auto kernel = std::dynamic_pointer_cast<stripe::Block>(main->stmts[0]);
  ApplyTile(kernel.get(), {2, 2, 2});
  ApplyCache(kernel, 0, "A");
  std::cout << *kernel;
}

}  // namespace test
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
