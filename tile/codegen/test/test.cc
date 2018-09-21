// Copyright 2018, Intel Corp.

#include <gmock/gmock.h>

#include "tile/codegen/tile.h"
#include "tile/codegen/vm.h"
#include "tile/lang/compose.h"
#include "tile/lang/gen_stripe.h"
#include "tile/stripe/stripe.h"

using ::testing::ContainerEq;
using ::testing::Eq;

namespace vertexai {
namespace tile {
namespace codegen {
namespace test {

lang::RunInfo LoadMatMul(size_t dim) {
  lang::RunInfo runinfo;
  runinfo.program_name = "matmul";
  runinfo.code = "function (A[M, K], B[K, N]) -> (C) { C[m, n : M, N] = +(A[m, k] * B[k, n]); }";
  runinfo.input_shapes.emplace("A", SimpleShape(DataType::FLOAT32, {dim, dim}));
  runinfo.input_shapes.emplace("B", SimpleShape(DataType::FLOAT32, {dim, dim}));
  runinfo.output_shapes.emplace("C", SimpleShape(DataType::FLOAT32, {dim, dim}));
  return runinfo;
}

lang::RunInfo LoadConv1D(size_t n, size_t x, size_t c, size_t k) {
  lang::RunInfo runinfo;
  runinfo.program_name = "conv1d";
  runinfo.code = R"(function (I[N, X, CI], K[KX, CI, CO]) -> (O) {
    O[n, x, co : N, X - KX + 1, CO] = +(I[n, x + k, ci] * K[k, ci, co]);
})";
  runinfo.input_shapes.emplace("I", SimpleShape(DataType::FLOAT32, {n, x, c}));
  runinfo.input_shapes.emplace("K", SimpleShape(DataType::FLOAT32, {k, c, c}));
  runinfo.output_shapes.emplace("O", SimpleShape(DataType::FLOAT32, {n, x - k + 1, c}));
  return runinfo;
}

lang::RunInfo LoadConv2D(size_t n, size_t x, size_t c, size_t k) {
  lang::RunInfo runinfo;
  runinfo.program_name = "conv2d";
  runinfo.code = R"(function (I[N, X, Y, CI], K[KX, KY, CI, CO]) -> (O) {
    O[n, x, y, co : N, X - KX + 1, Y - KY + 1, CO] = +(I[n, x + kx, y + ky, ci] * K[kx, ky, ci, co]);
})";
  runinfo.input_shapes.emplace("I", SimpleShape(DataType::FLOAT32, {n, x, x, c}));
  runinfo.input_shapes.emplace("K", SimpleShape(DataType::FLOAT32, {k, k, c, c}));
  runinfo.output_shapes.emplace("O", SimpleShape(DataType::FLOAT32, {n, x - k + 1, x - k + 1, c}));
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
  auto program = GenerateStripe(runinfo);

  IVLOG(2, "Before>\n" << program);

  ExecuteProgram(program, &data);

  IVLOG(2, "A: " << data["A"]);
  IVLOG(2, "B: " << data["B"]);
  IVLOG(2, "C: " << data["C"]);
  EXPECT_THAT(data["C"], ContainerEq(expected));

  auto main = std::dynamic_pointer_cast<stripe::Block>(program.stmts[0]);
  auto kernel = std::dynamic_pointer_cast<stripe::Block>(main->stmts[0]);
  ApplyTile(kernel.get(), {5, 4, 4});
  auto inner = std::dynamic_pointer_cast<stripe::Block>(kernel->stmts[0]);
  IVLOG(2, "Inner>\n" << *inner);
  ApplyTile(inner.get(), {5, 2, 2});

  for (size_t i = 0; i < data["C"].size(); i++) {
    data["C"][i] = 0;
  }

  IVLOG(2, "After>\n" << program);

  ExecuteProgram(program, &data);

  IVLOG(2, "A: " << data["A"]);
  IVLOG(2, "B: " << data["B"]);
  IVLOG(2, "C: " << data["C"]);
  EXPECT_THAT(data["C"], ContainerEq(expected));
}

TEST(Codegen, StencilMatchMatMul) {
  std::vector<StencilCriteria> criteria = {
      {"k", 16, {-1}, {-1, 0}},
      {"x", 16, {-1}, {0, -1}},
      {"c", -1, {0}, {-1, -1}},
  };

  auto runinfo = LoadMatMul(100);
  auto program = GenerateStripe(runinfo);
  auto main = std::dynamic_pointer_cast<stripe::Block>(program.stmts[0]);
  auto kernel = std::dynamic_pointer_cast<stripe::Block>(main->stmts[0]);

  IVLOG(2, *kernel);

  auto match = FindBestStencil({criteria}, kernel.get());
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
      {"k", 16, {-1}, {-1, 0}},
      {"x", 16, {-1}, {0, -1}},
      {"c", -1, {0}, {-1, -1}},
  };

  auto runinfo = LoadConv1D(1, 100, 64, 3);
  auto program = GenerateStripe(runinfo);
  auto main = std::dynamic_pointer_cast<stripe::Block>(program.stmts[0]);
  auto kernel = std::dynamic_pointer_cast<stripe::Block>(main->stmts[0]);

  IVLOG(2, *kernel);

  auto match = FindBestStencil({criteria}, kernel.get());
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
          {"k", 16, {-1}, {-1, 0}},  //
          {"x", 16, {-1}, {0, -1}},  //
          {"c", -1, {0}, {-1, -1}},  //
      },
      {
          {"k", 16, {-1}, {-1, 0}},  //
          {"x", 4, {-1}, {0, -1}},   //
          {"y", 4, {-1}, {0, -1}},   //
          {"c", -1, {0}, {-1, -1}}   //
      }                              //
  };

  auto runinfo = LoadConv2D(1, 100, 64, 3);
  auto program = GenerateStripe(runinfo);
  auto main = std::dynamic_pointer_cast<stripe::Block>(program.stmts[0]);
  auto kernel = std::dynamic_pointer_cast<stripe::Block>(main->stmts[0]);

  IVLOG(2, *kernel);

  auto match = FindBestStencil({criteria}, kernel.get());
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
      {"k", 16, {-1}, {-1, 0}},
      {"x", 16, {-1}, {0, -1}},
      {"c", -1, {0}, {-1, -1}},
  }};
  auto runinfo = LoadMatMul(100);
  auto program = GenerateStripe(runinfo);
  TilePass(&program, criteria);
  IVLOG(2, program);
}

TEST(Codegen, TilePassBroadcast) {
  lang::RunInfo runinfo;
  runinfo.program_name = "broadcast";
  runinfo.code = "function (A, B) -> (C) { C = add(A, B); }";
  runinfo.input_shapes.emplace("A", SimpleShape(DataType::FLOAT32, {1, 112, 112, 32}));
  runinfo.input_shapes.emplace("B", SimpleShape(DataType::FLOAT32, {32}));
  runinfo.output_shapes.emplace("C", SimpleShape(DataType::FLOAT32, {1, 112, 112, 32}));
  auto program = GenerateStripe(runinfo);

  LOG(INFO) << "\n" << program;
}

}  // namespace test
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
