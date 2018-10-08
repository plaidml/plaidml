// Copyright 2018, Intel Corp.

#include <gmock/gmock.h>

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

namespace {

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

}  // namespace

TEST(Codegen, Cache) {
  auto runinfo = LoadMatMul();
  auto program = GenerateStripe(runinfo);
  auto kernel = stripe::Block::Downcast(program->stmts.front());
  std::cout << "Original\n";
  std::cout << *program;
  ApplyTile(kernel.get(), {2, 2, 2});
  std::cout << "Tiled\n";
  std::cout << *program;
  ApplyCache(kernel.get(), "A");
  std::cout << "Cached\n";
  std::cout << *program;
}

}  // namespace test
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
