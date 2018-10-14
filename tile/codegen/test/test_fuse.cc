// Copyright 2018, Intel Corp.

#include <gmock/gmock.h>

#include "tile/codegen/fuse.h"
#include "tile/codegen/tile.h"
#include "tile/lang/compose.h"
#include "tile/lang/gen_stripe.h"
#include "tile/stripe/stripe.h"

using ::testing::ContainerEq;
using ::testing::Eq;

namespace vertexai {
namespace tile {
namespace codegen {
namespace test {

using namespace stripe;  // NOLINT

TEST(Codegen, FuseSimple) {
  lang::RunInfo runinfo;
  runinfo.program_name = "simple_fuse";
  runinfo.code = R"***(
    function (A, B) -> (C) { 
      T = A + B;
      C = (T < 0 ? 0 : T);
    }
  )***";
  runinfo.input_shapes.emplace("A", SimpleShape(DataType::FLOAT32, {100, 20}));
  runinfo.input_shapes.emplace("B", SimpleShape(DataType::FLOAT32, {20}));
  runinfo.output_shapes.emplace("C", SimpleShape(DataType::FLOAT32, {100, 20}));
  auto program = GenerateStripe(runinfo);
  auto main = Block::Downcast(program->stmts.front());

  IVLOG(2, "Before>\n" << *program);

  AliasMap base;
  AliasMap prog_map(base, *program);
  AliasMap main_map(prog_map, *main);

  auto k1 = Block::Downcast(*main->stmts.begin());
  auto k2 = Block::Downcast(*(++main->stmts.begin()));
  auto k3 = Block::Downcast(*(++++main->stmts.begin()));

  bool r = FuseBlocks(main_map, k1.get(), k2.get());
  r &= FuseBlocks(main_map, k1.get(), k3.get());
  IVLOG(2, "fuse r = " << r);
  IVLOG(2, "ki>\n" << *k1);

  ASSERT_TRUE(r);
}

TEST(Codegen, FuseComplex) {
  lang::RunInfo runinfo;
  runinfo.program_name = "complex_fuse";
  runinfo.code = R"***(
    function (In[N, X, Y, CI], K[I, J, CI, CO], B[CO]) -> (R) { 
      O[n, x, y, co : N, X, Y, CO] = +(In[n, x+i-1, y+j-1, ci] * K[i, j, ci, co]);
      BO = O + B;
      R = relu(BO);
    }
  )***";
  runinfo.input_shapes.emplace("In", SimpleShape(DataType::FLOAT32, {16, 100, 100, 64}));
  runinfo.input_shapes.emplace("K", SimpleShape(DataType::FLOAT32, {3, 3, 64, 128}));
  runinfo.input_shapes.emplace("B", SimpleShape(DataType::FLOAT32, {128}));
  runinfo.output_shapes.emplace("R", SimpleShape(DataType::FLOAT32, {16, 100, 100, 128}));
  auto program = GenerateStripe(runinfo);
  auto main = Block::Downcast(program->stmts.front());

  IVLOG(2, "Before>\n" << *program);

  AliasMap base;
  AliasMap prog_map(base, *program);
  AliasMap main_map(prog_map, *main);

  auto k1 = Block::Downcast(*main->stmts.begin());
  auto k2 = Block::Downcast(*(++main->stmts.begin()));
  auto plan = ComputeFusionPlan(*k1, *k2, "O");
  ASSERT_TRUE(static_cast<bool>(plan));
  auto r1 = FusionRefactor(*k1, plan->remap_a);
  auto r2 = FusionRefactor(*k2, plan->remap_b);
  IVLOG(2, "r1\n" << *r1);
  IVLOG(2, "r2\n" << *r2);
  bool r = FuseBlocks(main_map, r1.get(), r2.get());

  IVLOG(2, "Fused\n" << *r1);
  ASSERT_TRUE(r);

  // Tile it just for fun!
  ApplyTile(r1.get(), {16, 4, 4, 64}, "test");

  IVLOG(2, "Tiled\n" << *r1);
}

}  // namespace test
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
