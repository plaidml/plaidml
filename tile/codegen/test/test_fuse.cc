// Copyright 2018, Intel Corp.

#include <gmock/gmock.h>

#include "base/util/stream_container.h"
#include "tile/codegen/cache.h"
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

static std::shared_ptr<Block> SubBlock(size_t pos, const std::shared_ptr<Block>& block) {
  auto it = block->stmts.begin();
  for (size_t i = 0; i < pos; i++) {
    ++it;
  }
  return Block::Downcast(*it);
}

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
  auto main = SubBlock(0, program);

  IVLOG(2, "Before>\n" << *program);

  AliasMap base;
  AliasMap prog_map(base, *program);
  AliasMap main_map(prog_map, *main);

  auto k1 = SubBlock(0, main);
  auto k2 = SubBlock(1, main);
  auto k3 = SubBlock(2, main);

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
  auto main = SubBlock(0, program);

  IVLOG(2, "Before>\n" << *program);

  AliasMap base;
  AliasMap prog_map(base, *program);
  AliasMap main_map(prog_map, *main);

  auto k1 = SubBlock(0, main);
  auto k2 = SubBlock(1, main);
  auto plan = ComputeFusionPlan(*k1, *k2, "O");
  ASSERT_TRUE(static_cast<bool>(plan));
  auto r1 = FusionRefactor(*k1, plan->remap_a, plan->tile_a, "DPU");
  auto r2 = FusionRefactor(*k2, plan->remap_b, plan->tile_b, "DPU");
  IVLOG(2, "r1\n" << *r1);
  IVLOG(2, "r2\n" << *r2);
  bool r = FuseBlocks(main_map, r1.get(), r2.get());

  IVLOG(2, "Fused\n" << *r1);
  ASSERT_TRUE(r);

  // Tile it just for fun!
  ApplyTile(r1.get(), {16, 4, 4, 64}, "test", "location");

  IVLOG(2, "Tiled\n" << *r1);
}

TEST(Codegen, FuseTiled) {
  lang::RunInfo runinfo;
  runinfo.program_name = "tiled_fuse";
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
  auto main = SubBlock(0, program);

  // IVLOG(2, "Before>\n" << *program);

  AliasMap base;
  AliasMap prog_map(base, *program);
  AliasMap main_map(prog_map, *main);

  // Get the convolution
  auto k1 = SubBlock(0, main);
  // Tile it
  ApplyTile(k1.get(), {16, 16, 1, 1, 16, 1, 1}, "test", "location");
  // Get the bias add
  auto k2 = SubBlock(1, main);
  // Try to fuse it
  auto plan = ComputeFusionPlan(*k1, *k2, "O");
  IVLOG(2, "Plan as bool: " << static_cast<bool>(plan));
  IVLOG(2, "Remap a: " << StreamContainer(plan->remap_a));
  IVLOG(2, "Remap b: " << StreamContainer(plan->remap_b));
  IVLOG(2, "Tile a: " << StreamContainer(plan->tile_a));
  IVLOG(2, "Tile b: " << StreamContainer(plan->tile_b));
  // Refactor a, tile and refactor b, fuse
  ASSERT_TRUE(static_cast<bool>(plan));
  auto r1 = FusionRefactor(*k1, plan->remap_a, plan->tile_a, "DPU");
  auto r2 = FusionRefactor(*k2, plan->remap_b, plan->tile_b, "DPU");
  IVLOG(2, "r1\n" << *r1);
  IVLOG(2, "r2\n" << *r2);
  bool r = FuseBlocks(main_map, r1.get(), r2.get());
  IVLOG(2, "Fused\n" << *r1);

  // Now cache output for fun
  ApplyCache(r1.get(), "B", "CMX", "DMA");
  ApplyCache(r1.get(), "O", "CMX", "DMA");
  ApplyCache(r1.get(), "BO", "CMX", "DMA");
  auto inner = SubBlock(1, r1);
  IVLOG(1, "Inner\n" << *inner);
  ApplyCache(inner.get(), "In", "CMX", "DMA");
  ApplyCache(inner.get(), "K", "CMX", "DMA");
  IVLOG(2, "Fused + Cached\n" << *r1);

  ASSERT_TRUE(r);
}

TEST(Codegen, FuseFancy) {
  lang::RunInfo runinfo;
  runinfo.program_name = "tiled_fuse";
  runinfo.code = R"***(
    function (In[N, X, Y, CI], K1[I1, J1, CI1, CO1], K2[I2, J2, CI2, CO2]) -> (O2) { 
      O1[n, x, y, co : N, X, Y, CO1] = +(In[n, x+i-1, y+j-1, ci] * K1[i, j, ci, co]);
      O2[n, x, y, co : N, X, Y, CO2] = +(O1[n, x+i, y+j, ci] * K2[i, j, ci, co]);
    }
  )***";
  runinfo.input_shapes.emplace("In", SimpleShape(DataType::FLOAT32, {16, 100, 100, 64}));
  runinfo.input_shapes.emplace("K1", SimpleShape(DataType::FLOAT32, {3, 3, 64, 128}));
  runinfo.input_shapes.emplace("K2", SimpleShape(DataType::FLOAT32, {1, 1, 128, 128}));
  runinfo.output_shapes.emplace("O2", SimpleShape(DataType::FLOAT32, {16, 100, 100, 128}));
  auto program = GenerateStripe(runinfo);
  auto main = SubBlock(0, program);

  IVLOG(2, "Before>\n" << *program);

  AliasMap base;
  AliasMap prog_map(base, *program);
  AliasMap main_map(prog_map, *main);

  // Get the first convolution
  auto k1 = SubBlock(0, main);
  // Tile it
  ApplyTile(k1.get(), {16, 16, 1, 1, 16, 1, 1}, "test", "location");
  // Get the second convolution
  auto k2 = SubBlock(1, main);
  // Tile it as well
  ApplyTile(k2.get(), {16, 16, 16, 1, 1}, "test", "location");
  // Try to fuse it
  auto plan = ComputeFusionPlan(*k1, *k2, "O1");
  IVLOG(2, "Plan as bool: " << static_cast<bool>(plan));
  IVLOG(2, "Remap a: " << StreamContainer(plan->remap_a));
  IVLOG(2, "Remap b: " << StreamContainer(plan->remap_b));
  IVLOG(2, "Tile a: " << StreamContainer(plan->tile_a));
  IVLOG(2, "Tile b: " << StreamContainer(plan->tile_b));
  // Refactor a, tile and refactor b, fuse
  ASSERT_TRUE(static_cast<bool>(plan));
  auto r1 = FusionRefactor(*k1, plan->remap_a, plan->tile_a, "DPU");
  auto r2 = FusionRefactor(*k2, plan->remap_b, plan->tile_b, "DPU");
  IVLOG(2, "r1\n" << *r1);
  IVLOG(2, "r2\n" << *r2);
  bool r = FuseBlocks(main_map, r1.get(), r2.get());
  IVLOG(2, "Fused\n" << *r1);
  // Do some caching
  ApplyCache(r1.get(), "O1", "CMX", "DMA");
  IVLOG(2, "Cached\n" << *r1);

  ASSERT_TRUE(r);
}

}  // namespace test
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
