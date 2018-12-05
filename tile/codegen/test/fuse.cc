// Copyright 2018, Intel Corp.

#include <gmock/gmock.h>

#include "base/util/stream_container.h"
#include "testing/matchers.h"
#include "tile/codegen/cache.h"
#include "tile/codegen/fuse.h"
#include "tile/codegen/localize.h"
#include "tile/codegen/scalarize.h"
#include "tile/codegen/tile.h"
#include "tile/codegen/vm.h"
#include "tile/lang/compose.h"
#include "tile/lang/gen_stripe.h"
#include "tile/stripe/stripe.h"

using ::testing::ContainerEq;
using ::testing::Eq;
using ::testing::LinesEq;

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
      [[pid(add)]] T = A + B;
      [[pid(cmp_lt)]] X = T < 0;
      [[pid(cond)]] C = (X ? 0 : T);
    }
  )***";
  runinfo.input_shapes.emplace("A", SimpleShape(DataType::FLOAT32, {100, 20}));
  runinfo.input_shapes.emplace("B", SimpleShape(DataType::FLOAT32, {20}));
  runinfo.output_shapes.emplace("C", SimpleShape(DataType::FLOAT32, {100, 20}));
  auto program = GenerateStripe(runinfo);

  IVLOG(2, "Before>\n" << *program);

  proto::FusionPass pass1;
  pass1.add_a_reqs("eltwise_add");
  pass1.add_b_reqs("eltwise_cmp_lt");
  pass1.add_fused_set("fused");
  FusionPass(program.get(), pass1);

  proto::FusionPass pass2;
  pass2.add_a_reqs("fused");
  pass2.add_b_reqs("eltwise_cond");
  pass2.add_fused_set("fused");
  FusionPass(program.get(), pass2);

  IVLOG(2, "After>\n" << *program);

  auto expected = R"**(0: #program 
block []:1 ( // simple_fuse
    none new@0x00000000 A[0, 0] fp32(100, 20):(20, 1):7.8125 KiB
    none new@0x00000000 B[0] fp32(20):(1):80 B
    none new@0x00000000 C[0, 0] fp32(100, 20):(20, 1):7.8125 KiB
) {
  0: #main 
  block []:1 ( // main
      in A[0, 0] fp32(100, 20):(20, 1):7.8125 KiB
      in B[0] fp32(20):(1):80 B
      out C[0, 0]:assign fp32(100, 20):(20, 1):7.8125 KiB
      none new@0x00000000 T[0, 0] fp32(100, 20):(20, 1):7.8125 KiB
      none new@0x00000000 X[0, 0] bool(100, 20):(20, 1):1.95312 KiB
  ) {
    0: #fused 
    block [i1:100, i2:20]:2000 ( // add+cmp_lt+cond
        in A[i1, i2] fp32(1, 1):(20, 1):4 B
        in B[i2] fp32(1):(1):4 B
        out C[i1, i2] fp32(1, 1):(20, 1):4 B
        out T[i1, i2] fp32(1, 1):(20, 1):4 B
        out X[i1, i2] bool(1, 1):(20, 1):1 B
    ) {
      0: $A = load(A)
      1: $B = load(B)
      2: $T = add($A, $B)
      3: T = store($T)
      4: $T_0 = load(T)
      5: $_T1 = (int)0
      6: $X = cmp_lt($T_0, $_T1)
      7: X = store($X)
      8: $X_0 = load(X)
      9: $_T3 = (int)0
      10: $T_1 = load(T)
      11: $C = cond($X_0, $_T3, $T_1)
      12: C = store($C)
    }
  }
}
)**";
  EXPECT_THAT(to_string(*program), LinesEq(expected));
}

TEST(Codegen, FuseComplex) {
  std::map<std::string, std::vector<float>> data = {
      {"In",
       {
           1, 2, 3, 4,  //
           5, 4, 5, 6,  //
           7, 8, 7, 8,  //
           9, 7, 8, 7,  //
           8, 9, 7, 8,  //
       }},
      {"K",
       {
           1, 2, 3, 1,  //
           1, 2, 3, 1,  //
           1, 2, 3, 1,  //
           1, 2, 3, 1,  //
                        //
           1, 2, 3, 1,  //
           1, 2, 3, 1,  //
           1, 2, 3, 1,  //
           1, 2, 3, 1,  //
                        //
           1, 2, 3, 1,  //
           1, 2, 3, 1,  //
           1, 2, 3, 1,  //
           1, 2, 3, 1,  //
       }},
      {"B",
       {
           5, 6, 7, 8,  //
       }},
      {"R",
       {
           0, 0, 0, 0,  //
           0, 0, 0, 0,  //
           0, 0, 0, 0,  //
           0, 0, 0, 0,  //
           0, 0, 0, 0,  //
       }},
  };

  std::vector<float> expected = {
      35, 66,  97,  38,   //
      65, 126, 187, 68,   //
      86, 168, 250, 89,   //
      98, 192, 286, 101,  //
      68, 132, 196, 71,   //
  };

  lang::RunInfo runinfo;
  runinfo.program_name = "complex_fuse";
  runinfo.code = R"***(
    function (In[X, CI], K[I, CI, CO], B[CO]) -> (R) {
      [[pid(conv)]]     O[x, co : X, CO] = +(In[x+i-1, ci] * K[i, ci, co]);
      [[pid(bias_add)]] BO = O + B;
      [[pid(relu)]]     R = relu(BO);
    }
  )***";
  const size_t X = 5;
  const size_t K = 3;
  const size_t CI = 4;
  const size_t CO = 4;
  runinfo.input_shapes.emplace("In", SimpleShape(DataType::FLOAT32, {X, CI}));
  runinfo.input_shapes.emplace("K", SimpleShape(DataType::FLOAT32, {K, CI, CO}));
  runinfo.input_shapes.emplace("B", SimpleShape(DataType::FLOAT32, {CO}));
  runinfo.output_shapes.emplace("R", SimpleShape(DataType::FLOAT32, {X, CO}));
  auto program = GenerateStripe(runinfo);
  auto main = program->SubBlock(0);

  IVLOG(2, "Before>\n" << *program);
  ExecuteProgram(*program, &data);
  IVLOG(2, "R: " << data["R"]);
  EXPECT_THAT(data["R"], ContainerEq(expected));

  AliasMap base;
  AliasMap prog_map(base, program.get());
  AliasMap main_map(prog_map, main.get());

  AlwaysFuseRecursive afr;
  FusionInner(main_map, main.get(), &afr);
  // LocalizePass(main_map, main.get());
  Scalarize(main.get(), true);

  IVLOG(2, "After>\n" << *program);
  ExecuteProgram(*program, &data);
  IVLOG(2, "R: " << data["R"]);
  EXPECT_THAT(data["R"], ContainerEq(expected));
}

TEST(Codegen, FuseTiled) {
  lang::RunInfo runinfo;
  runinfo.program_name = "tiled_fuse";
  runinfo.code = R"***(
    function (In[N, X, Y, CI], K[I, J, CI, CO], B[CO]) -> (R) { 
      [[pid(conv)]]     O[n, x, y, co : N, X, Y, CO] = +(In[n, x+i-1, y+j-1, ci] * K[i, j, ci, co]);
      [[pid(bias_add)]] BO = O + B;
      [[pid(relu)]]     R = relu(BO);
    }
  )***";
  runinfo.input_shapes.emplace("In", SimpleShape(DataType::FLOAT32, {16, 100, 100, 64}));
  runinfo.input_shapes.emplace("K", SimpleShape(DataType::FLOAT32, {3, 3, 64, 128}));
  runinfo.input_shapes.emplace("B", SimpleShape(DataType::FLOAT32, {128}));
  runinfo.output_shapes.emplace("R", SimpleShape(DataType::FLOAT32, {16, 100, 100, 128}));
  auto program = GenerateStripe(runinfo);
  auto main = program->SubBlock(0);

  // IVLOG(2, "Before>\n" << *program);

  AliasMap base;
  AliasMap prog_map(base, program.get());
  AliasMap main_map(prog_map, main.get());

  // Get the convolution
  auto k1 = main->SubBlock(0);
  // Tile it
  ApplyTile(k1.get(), {16, 16, 1, 1, 16, 1, 1});
  // Get the bias add
  auto k2 = main->SubBlock(1);
  // Try to fuse it
  auto plan = ComputeFusionPlan(*k1, *k2, "O");
  IVLOG(2, "Plan as bool: " << static_cast<bool>(plan));
  IVLOG(2, "Remap a: " << StreamContainer(plan->remap_a));
  IVLOG(2, "Remap b: " << StreamContainer(plan->remap_b));
  IVLOG(2, "Tile a: " << StreamContainer(plan->tile_a));
  IVLOG(2, "Tile b: " << StreamContainer(plan->tile_b));
  // Refactor a, tile and refactor b, fuse
  ASSERT_TRUE(static_cast<bool>(plan));
  auto r1 = FusionRefactor(*k1, plan->remap_a, plan->tile_a);
  auto r2 = FusionRefactor(*k2, plan->remap_b, plan->tile_b);
  IVLOG(2, "r1\n" << *r1);
  IVLOG(2, "r2\n" << *r2);
  bool r = FuseBlocks(main_map, r1.get(), r2.get());
  ASSERT_TRUE(r);
  IVLOG(2, "Fused\n" << *r1);

  // Now cache output for fun
  ApplyCache(r1.get(), "B", {"CMX"}, {"DMA"});
  ApplyCache(r1.get(), "O", {"CMX"}, {"DMA"});
  ApplyCache(r1.get(), "BO", {"CMX"}, {"DMA"});
  IVLOG(1, "Cached\n" << *program);

  auto inner = r1->SubBlock(1);
  IVLOG(1, "Inner\n" << *inner);
  ApplyCache(inner.get(), "In", {"CMX"}, {"DMA"});
  ApplyCache(inner.get(), "K", {"CMX"}, {"DMA"});
  IVLOG(2, "Fused + Cached\n" << *r1);
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
  auto main = program->SubBlock(0);

  IVLOG(2, "Before>\n" << *program);

  AliasMap base;
  AliasMap prog_map(base, program.get());
  AliasMap main_map(prog_map, main.get());

  // Get the first convolution
  auto k1 = main->SubBlock(0);
  // Tile it
  ApplyTile(k1.get(), {16, 16, 1, 1, 16, 1, 1});
  // Get the second convolution
  auto k2 = main->SubBlock(1);
  // Tile it as well
  ApplyTile(k2.get(), {16, 16, 16, 1, 1});
  // Try to fuse it
  auto plan = ComputeFusionPlan(*k1, *k2, "O1");
  IVLOG(2, "Plan as bool: " << static_cast<bool>(plan));
  IVLOG(2, "Remap a: " << StreamContainer(plan->remap_a));
  IVLOG(2, "Remap b: " << StreamContainer(plan->remap_b));
  IVLOG(2, "Tile a: " << StreamContainer(plan->tile_a));
  IVLOG(2, "Tile b: " << StreamContainer(plan->tile_b));
  // Refactor a, tile and refactor b, fuse
  ASSERT_TRUE(static_cast<bool>(plan));
  auto r1 = FusionRefactor(*k1, plan->remap_a, plan->tile_a);
  auto r2 = FusionRefactor(*k2, plan->remap_b, plan->tile_b);
  IVLOG(2, "r1\n" << *r1);
  IVLOG(2, "r2\n" << *r2);
  bool r = FuseBlocks(main_map, r1.get(), r2.get());
  IVLOG(2, "Fused\n" << *r1);
  // Do some caching
  ApplyCache(r1.get(), "O1", {"CMX"}, {"DMA"});
  IVLOG(2, "Cached\n" << *r1);

  ASSERT_TRUE(r);
}

TEST(Codegen, FuseAuto) {
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
  auto main = program->SubBlock(0);

  // Get the convolution + tile it
  auto k1 = main->SubBlock(0);
  ApplyTile(k1.get(), {16, 16, 1, 1, 16, 1, 1});

  IVLOG(2, "Before>\n" << *program);

  AliasMap base;
  AliasMap prog_map(base, program.get());
  AliasMap main_map(prog_map, main.get());

  AlwaysFuseRecursive afr;
  FusionInner(main_map, main.get(), &afr);
  LocalizePass(main_map, main.get());
  Scalarize(main.get(), true);

  IVLOG(2, "After>\n" << *program);
}

}  // namespace test
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
