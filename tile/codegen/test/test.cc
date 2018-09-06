// Copyright 2018, Intel Corp.

#include <gmock/gmock.h>

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

lang::RunInfo LoadMatMul() {
  const size_t DIM = 5;
  lang::RunInfo runinfo;
  runinfo.code = "function (A[M, K], B[K, N]) -> (C) { C[m, n : M, N] = +(A[m, k] * B[k, n]); }";
  runinfo.input_shapes.emplace("A", lang::SimpleShape(lang::DataType::FLOAT32, {DIM, DIM}));
  runinfo.input_shapes.emplace("B", lang::SimpleShape(lang::DataType::FLOAT32, {DIM, DIM}));
  runinfo.output_shapes.emplace("C", lang::SimpleShape(lang::DataType::FLOAT32, {DIM, DIM}));
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

TEST(Codegen, Basic) {
  auto runinfo = LoadMatMul();
  auto program = GenerateStripe("matmul", runinfo);

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

  std::cout << "Before>" << std::endl << program << std::endl;

  ExecuteProgram(program, &data);

  std::cout << "A: " << data["A"] << std::endl;
  std::cout << "B: " << data["B"] << std::endl;
  std::cout << "C: " << data["C"] << std::endl;
  EXPECT_THAT(data["C"], Eq(expected));

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
  EXPECT_THAT(data["C"], Eq(expected));

  // Stencils:
  // k=16, x=16, c=?
  // k=16, x=4, y=4, c=?

  // Criteria:
  // k: 16  C:  1  A: !0  B:  0
  // x: 16  C: !0  A:  0  B: !0
  // c: ??  C:  0  A:  1  B:  1
  // or
  // k: 16  C:  1  A: !0  B:  0
  // x:  4  C: !0  A:  0  B: !0
  // y:  4  C: !0  A:  0  B: !0
  // c: ??  C:  0  A:  1  B:  1
}

}  // namespace test
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
