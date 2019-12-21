// Copyright 2018, Intel Corp.

#include <gmock/gmock.h>
#include <google/protobuf/util/json_util.h>

#include "plaidml2/edsl/helper.h"
#include "tile/codegen/stencil.h"
#include "tile/codegen/tile.h"
#include "tile/codegen/vm.h"
#include "tile/lang/gen_stripe.h"
#include "tile/lang/runinfo.h"
#include "tile/lib/lib.h"
#include "tile/stripe/stripe.h"

using ::testing::ContainerEq;
using ::testing::Eq;

namespace vertexai {
namespace tile {
namespace codegen {
namespace test {

namespace {

using plaidml2::edsl::LogicalShape;

template <typename T>
T ParseProtoJson(const std::string& str) {
  T proto;
  google::protobuf::util::JsonStringToMessage(str, &proto);
  return proto;
}

std::map<std::string, std::vector<float>> MakeMatMulTestData() {
  std::map<std::string, std::vector<float>> data = {
      {"A",
       {
           1, 2, 3, 4, 5,  //
           4, 5, 6, 7, 8,  //
           7, 8, 9, 7, 8,  //
           1, 2, 3, 1, 2,  //
           1, 2, 3, 1, 2,  //
       }},
      {"B",
       {
           1, 2, 3, 1, 2,  //
           1, 2, 3, 1, 2,  //
           1, 2, 3, 1, 2,  //
           1, 2, 3, 1, 2,  //
           1, 2, 3, 1, 2,  //
       }},
      {"C",
       {
           0, 0, 0, 0, 0,  //
           0, 0, 0, 0, 0,  //
           0, 0, 0, 0, 0,  //
           0, 0, 0, 0, 0,  //
           0, 0, 0, 0, 0,  //
       }},
  };
  return data;
}

std::vector<float> kMatMulExpected = {
    15, 30, 45,  15, 30,  //
    30, 60, 90,  30, 60,  //
    39, 78, 117, 39, 78,  //
    9,  18, 27,  9,  18,  //
    9,  18, 27,  9,  18,  //
};

std::vector<float> kConv1dExpected = {
    45, 90,  135, 45, 90,   //
    84, 168, 252, 84, 168,  //
    78, 156, 234, 78, 156,  //
    57, 114, 171, 57, 114,  //
    18, 36,  54,  18, 36,   //
};

}  // namespace

TEST(Codegen, ApplyTile) {
  int64_t dim = sqrt(kMatMulExpected.size());
  auto tileProgram = lib::LoadMatMul(                  //
      "matmul",                                        //
      LogicalShape(PLAIDML_DATA_FLOAT32, {dim, dim}),  //
      LogicalShape(PLAIDML_DATA_FLOAT32, {dim, dim}));
  auto program = plaidml2::edsl::ConvertIntoStripe(tileProgram);
  auto main = program->entry->SubBlock(0);
  auto data = MakeMatMulTestData();

  IVLOG(2, "Before>\n" << *program->entry);

  ExecuteProgram(*program->entry, &data);

  IVLOG(2, "A: " << data["A"]);
  IVLOG(2, "B: " << data["B"]);
  IVLOG(2, "C: " << data["C"]);
  EXPECT_THAT(data["C"], ContainerEq(kMatMulExpected));

  auto kernel = main->SubBlock(0);
  ApplyTile(kernel.get(), {5, 2, 2});
  auto inner = kernel->SubBlock(0);
  inner->name = "inner";
  ApplyTile(inner.get(), {5, 3, 2, 1, 1});
  auto innermost = inner->SubBlock(0);
  innermost->name = "innermost";
  ApplyTile(kernel.get(), {5, 2, 2});
  kernel->SubBlock(0)->name = "outer";

  IVLOG(2, "After>\n" << *program->entry);

  std::fill(data["C"].begin(), data["C"].end(), 0.f);
  ExecuteProgram(*program->entry, &data);

  IVLOG(2, "A: " << data["A"]);
  IVLOG(2, "B: " << data["B"]);
  IVLOG(2, "C: " << data["C"]);
  EXPECT_THAT(data["C"], ContainerEq(kMatMulExpected));
}

TEST(Stencil, MatchMatMul) {
  auto spec = ParseProtoJson<proto::Stencil>(R"(
    {
      "startup_cost": 32,
      "idxs": [
        { "name": "k", "size": 16, "outs": [ -1 ], "ins": [ -1,  0 ] },
        { "name": "x", "size": 16, "outs": [ -1 ], "ins": [  0, -1 ] },
        { "name": "c", "size": -1, "outs": [  0 ], "ins": [ -1, -1 ] },
      ]
    }
  )");

  const std::int64_t DIM = 100;
  auto tileProgram = lib::LoadMatMul(                  //
      "matmul",                                        //
      LogicalShape(PLAIDML_DATA_FLOAT32, {DIM, DIM}),  //
      LogicalShape(PLAIDML_DATA_FLOAT32, {DIM, DIM}));
  auto program = plaidml2::edsl::ConvertIntoStripe(tileProgram);
  auto main = program->entry->SubBlock(0);
  auto kernel = main->SubBlock(0);

  IVLOG(2, *kernel);

  auto match = FindBestStencil({spec}, false, kernel.get());
  ASSERT_TRUE(match);
  IVLOG(1, "Best match: " << *match);
  StencilMatch expected{
      1255968,  // total
      false,
      {
          {"k", "c", 100},
          {"m", "k", 16},
          {"n", "x", 16},
      }  // idxs
  };
  EXPECT_THAT(*match, Eq(expected));
}

TEST(Stencil, MatchMatMulForXSMMStrict) {
  auto spec = ParseProtoJson<proto::Stencil>(R"(
    {
      startup_cost: 32,
      idxs: [
        { "name": "m", "size": 64, "outs": [1], "ins": [1, 0] },
        { "name": "n", "size": 16, "outs": [-1], "ins": [0, -1] },
        { "name": "k", "size": 64, "outs": [0], "ins": [-1, 1] },
      ],
    }
  )");

  const std::int64_t DIM = 1024;
  auto tileProgram = lib::LoadMatMul(                  //
      "matmul",                                        //
      LogicalShape(PLAIDML_DATA_FLOAT32, {DIM, DIM}),  //
      LogicalShape(PLAIDML_DATA_FLOAT32, {DIM, DIM}));
  auto program = plaidml2::edsl::ConvertIntoStripe(tileProgram);
  auto main = program->entry->SubBlock(0);
  auto kernel = main->SubBlock(0);

  IVLOG(2, *kernel);

  auto match = FindBestStencil({spec}, true, kernel.get());
  ASSERT_TRUE(match);
  IVLOG(1, "Best match: " << *match);
  StencilMatch expected{
      1074266112,  // total
      false,
      {
          {"k", "k", 64},
          {"m", "n", 16},
          {"n", "m", 64},
      }  // idxs
  };

  EXPECT_THAT(*match, Eq(expected));
}

TEST(Stencil, MatchMatMulForXSMMNonStrict) {
  auto spec = ParseProtoJson<proto::Stencil>(R"(
    {
      startup_cost: 32,
      idxs: [
        { name: 'm', size: 64, outs: [1], ins: [1, 0] },
        { name: 'n', size: 16, outs: [-1], ins: [0, -1] },
        { name: 'k', size: 64, outs: [0], ins: [-1, 1] },
      ],
    }
  )");

  const std::int64_t DIM = 100;
  auto tileProgram = lib::LoadMatMul(                  //
      "matmul",                                        //
      LogicalShape(PLAIDML_DATA_FLOAT32, {DIM, DIM}),  //
      LogicalShape(PLAIDML_DATA_FLOAT32, {DIM, DIM}));
  auto program = plaidml2::edsl::ConvertIntoStripe(tileProgram);
  auto main = program->entry->SubBlock(0);
  auto kernel = main->SubBlock(0);

  IVLOG(2, *kernel);

  auto match = FindBestStencil({spec}, true, kernel.get());
  ASSERT_FALSE(match);
}

TEST(Stencil, MatchConv1D) {
  auto spec = ParseProtoJson<proto::Stencil>(R"(
    {
      "startup_cost": 32,
      "idxs": [
        { "name": "k", "size": 16, "outs": [ -1 ], "ins": [ -1,  0 ] },
        { "name": "x", "size": 16, "outs": [ -1 ], "ins": [  0, -1 ] },
        { "name": "c", "size": -1, "outs": [  0 ], "ins": [ -1, -1 ] },
      ]
    }
  )");

  auto tileProgram = lib::LoadConv1d(                    //
      "conv",                                            //
      LogicalShape(PLAIDML_DATA_FLOAT32, {1, 100, 64}),  //
      LogicalShape(PLAIDML_DATA_FLOAT32, {3, 64, 64}),   //
      {1, 100, 64});
  auto program = plaidml2::edsl::ConvertIntoStripe(tileProgram);
  auto main = program->entry->SubBlock(0);
  auto kernel = main->SubBlock(0);

  IVLOG(2, *kernel);

  auto match = FindBestStencil({spec}, false, kernel.get());
  ASSERT_TRUE(match);
  IVLOG(1, "Best match: " << *match);
  StencilMatch expected{
      1378944,  // total
      false,
      {
          {"ci", "c", 64},
          {"co", "k", 16},
          {"k0", "*", 1},
          {"n", "*", 1},
          {"x0", "x", 16},
      }  // idxs
  };
  EXPECT_THAT(*match, Eq(expected));
}

TEST(Stencil, MatchConv2D) {
  auto options = ParseProtoJson<proto::StencilPass>(R"(
    {
      "stencils": [
        {
          "startup_cost": 32,
          "idxs": [
            { "name": "k", "size": 16, "outs": [ -1 ], "ins": [ -1,  0 ] },
            { "name": "x", "size": 16, "outs": [ -1 ], "ins": [  0, -1 ] },
            { "name": "c", "size": -1, "outs": [  0 ], "ins": [ -1, -1 ] },
          ]
        },
        {
          "startup_cost": 32,
          "idxs": [
            { "name": "k", "size": 16, "outs": [ -1 ], "ins": [ -1,  0 ] },
            { "name": "x", "size":  4, "outs": [ -1 ], "ins": [  0, -1 ] },
            { "name": "y", "size":  4, "outs": [ -1 ], "ins": [  0, -1 ] },
            { "name": "c", "size": -1, "outs": [  0 ], "ins": [ -1, -1 ] },
          ]
        }
      ]
    }
  )");
  std::vector<proto::Stencil> specs;
  for (const auto& stencil : options.stencils()) {
    specs.push_back(stencil);
  }

  auto tileProgram = lib::LoadConv2d(                         //
      "conv",                                                 //
      LogicalShape(PLAIDML_DATA_FLOAT32, {1, 100, 100, 56}),  //
      LogicalShape(PLAIDML_DATA_FLOAT32, {3, 3, 56, 56}),     //
      {1, 100, 100, 56});                                     //
  auto program = plaidml2::edsl::ConvertIntoStripe(tileProgram);
  auto main = program->entry->SubBlock(0);
  auto kernel = main->SubBlock(0);

  IVLOG(2, "\n" << *kernel);

  auto match = FindBestStencil(specs, false, kernel.get());
  ASSERT_TRUE(match);
  IVLOG(1, "Best match: " << *match);
  StencilMatch expected{
      323280000,  // total
      false,
      {
          {"ci", "c", 56},
          {"co", "k", 16},
          {"k0", "*", 1},
          {"k1", "*", 1},
          {"n", "*", 1},
          {"x0", "x", 4},
          {"x1", "y", 4},
      }  // idxs
  };
  EXPECT_THAT(*match, Eq(expected));
}

TEST(Stencil, Pass) {
  auto options = ParseProtoJson<proto::StencilPass>(R"(
    {
      "reqs": ["agg_op_add", "comb_op_mul"],
      "startup_cost": 32,
      "idxs": [
        { "name": "k", "size":  2, "outs": [ -1 ], "ins": [ -1,  0 ] },
        { "name": "x", "size":  2, "outs": [ -1 ], "ins": [  0, -1 ] },
        { "name": "c", "size": -1, "outs": [  0 ], "ins": [ -1, -1 ] },
      ]
    }
  )");

  const std::int64_t DIM = 5;
  auto tileProgram = lib::LoadMatMul(                  //
      "matmul",                                        //
      LogicalShape(PLAIDML_DATA_FLOAT32, {DIM, DIM}),  //
      LogicalShape(PLAIDML_DATA_FLOAT32, {DIM, DIM}));
  auto program = plaidml2::edsl::ConvertIntoStripe(tileProgram);
  CompilerState state(program);
  StencilPass(options).Apply(&state);
  IVLOG(2, "\n" << *program->entry);

  auto data = MakeMatMulTestData();
  ExecuteProgram(*program->entry, &data);

  IVLOG(2, "A: " << data["A"]);
  IVLOG(2, "B: " << data["B"]);
  IVLOG(2, "C: " << data["C"]);
  EXPECT_THAT(data["C"], ContainerEq(kMatMulExpected));
}

}  // namespace test
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
