// Copyright 2018, Intel Corp.

#include <gmock/gmock.h>

#include "tile/codegen/tile.h"
#include "tile/codegen/vm.h"
#include "tile/lang/compose.h"
#include "tile/lang/gen_stripe.h"
#include "tile/lang/stripe.h"

using ::testing::ContainerEq;
using ::testing::Eq;

namespace vertexai {
namespace tile {
namespace codegen {
namespace test {

lang::RunInfo LoadMatMul(size_t dim) {
  lang::RunInfo runinfo;
  runinfo.code = "function (A[M, K], B[K, N]) -> (C) { C[m, n : M, N] = +(A[m, k] * B[k, n]); }";
  runinfo.input_shapes.emplace("A", lang::SimpleShape(lang::DataType::FLOAT32, {dim, dim}));
  runinfo.input_shapes.emplace("B", lang::SimpleShape(lang::DataType::FLOAT32, {dim, dim}));
  runinfo.output_shapes.emplace("C", lang::SimpleShape(lang::DataType::FLOAT32, {dim, dim}));
  return runinfo;
}

lang::RunInfo LoadConv1D(size_t n, size_t x, size_t c, size_t k) {
  lang::RunInfo runinfo;
  runinfo.code = R"(function (I[N, X, CI], K[KX, CI, CO]) -> (O) {
    O[n, x, co : N, X - KX + 1, CO] = +(I[n, x + k, ci] * K[k, ci, co]);
})";
  runinfo.input_shapes.emplace("I", lang::SimpleShape(lang::DataType::FLOAT32, {n, x, c}));
  runinfo.input_shapes.emplace("K", lang::SimpleShape(lang::DataType::FLOAT32, {k, c, c}));
  runinfo.output_shapes.emplace("O", lang::SimpleShape(lang::DataType::FLOAT32, {n, x - k + 1, c}));
  return runinfo;
}

lang::RunInfo LoadConv2D(size_t n, size_t x, size_t c, size_t k) {
  lang::RunInfo runinfo;
  runinfo.code = R"(function (I[N, X, Y, CI], K[KX, KY, CI, CO]) -> (O) {
    O[n, x, y, co : N, X - KX + 1, Y - KY + 1, CO] = +(I[n, x + kx, y + ky, ci] * K[kx, ky, ci, co]);
})";
  runinfo.input_shapes.emplace("I", lang::SimpleShape(lang::DataType::FLOAT32, {n, x, x, c}));
  runinfo.input_shapes.emplace("K", lang::SimpleShape(lang::DataType::FLOAT32, {k, k, c, c}));
  runinfo.output_shapes.emplace("O", lang::SimpleShape(lang::DataType::FLOAT32, {n, x - k + 1, x - k + 1, c}));
  return runinfo;
}

std::ostream& operator<<(std::ostream& os, const std::vector<float>& data) {
  bool first = true;
  for (const auto& x : data) {
    if (!first) {
      os << ", ";
    }
    os << x;
    first = false;
  }
  return os;
}

TEST(Codegen, ApplyTile) {
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

  std::vector<float> expected = {
      15, 30, 45,  15, 30,  //
      30, 60, 90,  30, 60,  //
      39, 78, 117, 39, 78,  //
      9,  18, 27,  9,  18,  //
      9,  18, 27,  9,  18,  //
  };

  auto runinfo = LoadMatMul(sqrt(expected.size()));
  auto program = GenerateStripe("matmul", runinfo);

  std::cout << "Before>" << std::endl << program << std::endl;

  ExecuteProgram(program, &data);

  std::cout << "A: " << data["A"] << std::endl;
  std::cout << "B: " << data["B"] << std::endl;
  std::cout << "C: " << data["C"] << std::endl;
  EXPECT_THAT(data["C"], ContainerEq(expected));

  auto main = program.mutable_stmts(0)->mutable_block();
  auto kernel = main->mutable_stmts(0)->mutable_block();
  ApplyTile(kernel, {5, 4, 4});
  auto inner = kernel->mutable_stmts(0)->mutable_block();
  std::cout << "Inner>" << std::endl << *inner << std::endl;
  ApplyTile(inner, {5, 2, 2});

  for (size_t i = 0; i < data["C"].size(); i++) {
    data["C"][i] = 0;
  }

  std::cout << "After>" << std::endl << program << std::endl;

  ExecuteProgram(program, &data);

  std::cout << "A: " << data["A"] << std::endl;
  std::cout << "B: " << data["B"] << std::endl;
  std::cout << "C: " << data["C"] << std::endl;
  EXPECT_THAT(data["C"], ContainerEq(expected));
}

TEST(Codegen, StencilMatchMatMul) {
  std::vector<StencilCriteria> criteria = {
      {"k", 16, {-1, -1, 0}},
      {"x", 16, {-1, 0, -1}},
      {"c", -1, {0, -1, -1}},
  };

  auto runinfo = LoadMatMul(100);
  auto program = GenerateStripe("matmul", runinfo);
  auto main = program.stmts(0).block();
  auto kernel = main.stmts(0).block();

  std::cout << kernel << std::endl;

  auto match = FindBestStencil({criteria}, kernel);
  LOG(INFO) << "Best match: " << match;
  StencilMatch expected{
      25600,            // total
      {"c", "k", "x"},  // names
      {100, 16, 16},    // tile
  };
  EXPECT_THAT(match, Eq(expected));
}

TEST(Codegen, StencilMatchConv1D) {
  std::vector<StencilCriteria> criteria = {
      {"k", 16, {-1, -1, 0}},
      {"x", 16, {-1, 0, -1}},
      {"c", -1, {0, -1, -1}},
  };

  auto runinfo = LoadConv1D(1, 100, 64, 3);
  auto program = GenerateStripe("conv1d", runinfo);
  auto main = program.stmts(0).block();
  auto kernel = main.stmts(0).block();

  std::cout << kernel << std::endl;

  auto match = FindBestStencil({criteria}, kernel);
  LOG(INFO) << "Best match: " << match;
  StencilMatch expected{
      16384,                 // total
      {"c", "x", "*", "k"},  // names
      {64, 16, 1, 16},       // tile
  };
  EXPECT_THAT(match, Eq(expected));
}

TEST(Codegen, StencilMatchConv2D) {
  std::vector<std::vector<StencilCriteria>> criteria = {
      {
          {"k", 16, {-1, -1, 0}},  //
          {"x", 16, {-1, 0, -1}},  //
          {"c", -1, {0, -1, -1}},  //
      },
      {
          {"k", 16, {-1, -1, 0}},  //
          {"x", 4, {-1, 0, -1}},   //
          {"y", 4, {-1, 0, -1}},   //
          {"c", -1, {0, -1, -1}}   //
      }                            //
  };

  auto runinfo = LoadConv2D(1, 100, 64, 3);
  auto program = GenerateStripe("conv2d", runinfo);
  auto main = program.stmts(0).block();
  auto kernel = main.stmts(0).block();

  std::cout << kernel << std::endl;

  auto match = FindBestStencil({criteria}, kernel);
  LOG(INFO) << "Best match: " << match;
  StencilMatch expected{
      16384,                           // total
      {"c", "x", "*", "*", "k", "*"},  // names
      {64, 16, 1, 1, 16, 1},           // tile
  };
  EXPECT_THAT(match, Eq(expected));
}

TEST(Codegen, TilePass) {
  std::vector<std::vector<StencilCriteria>> criteria = {{
      {"k", 16, {-1, -1, 0}},
      {"x", 16, {-1, 0, -1}},
      {"c", -1, {0, -1, -1}},
  }};
  auto runinfo = LoadMatMul(100);
  auto program = GenerateStripe("matmul", runinfo);
  TilePass(&program, criteria);
  std::cout << program << std::endl;
}

}  // namespace test
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
