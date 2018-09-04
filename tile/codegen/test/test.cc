// Copyright 2018, Intel Corp.

#include <gtest/gtest.h>

#include "tile/codegen/tile.h"
#include "tile/codegen/vm.h"
#include "tile/lang/compose.h"
#include "tile/lang/gen_stripe.h"
#include "tile/lang/intrinsics.h"
#include "tile/lang/stripe.h"

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

  std::map<std::string, float*> buffers = {
      {"A", data["A"].data()},  //
      {"B", data["B"].data()},
      {"C", data["C"].data()},
  };

  // ExecuteProgram(program, buffers);

  // std::cout << "A: " << data["A"] << std::endl;
  // std::cout << "B: " << data["B"] << std::endl;
  // std::cout << "C: " << data["C"] << std::endl;

  std::cout << "Before>" << std::endl;
  lang::Print(std::cout, program);

  auto main = program.mutable_stmts(0)->mutable_block();
  auto kernel = main->mutable_stmts(0)->mutable_block();
  ApplyTile(kernel, {5, 2, 2});

  std::cout << "After>" << std::endl;
  lang::Print(std::cout, program);

  // for (int m = 0; m < M; m++) {
  //   for (int n = 0; n < N; n++) {
  //     float acc = 0.0f;
  //     for (int k = 0; k < K; k++) {
  //       acc += A[k * M + m] * B[n * K + k];
  //     }
  //     C[n * M + m] = acc;
  //   }
  // }

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

  // Transformation:
  // original
  // --------
  // block [] // $matmul
  //     () -> () {
  //   var A : FLOAT32[100:100, 100:1]
  //   var B : FLOAT32[100:100, 100:1]
  //   var C : FLOAT32[100:100, 100:1]
  //   block [] // main
  //       (A, B) -> (C:assign) {
  //     block [k:100, m:100, n:100] // kernel_0
  //         // C[m, n : M, N] = +(A[m, k] * B[k, n])
  //         (A[k + 100*m], B[100*k + n]) -> (C[100*m + n]:sum) {
  //       $A = load(A)
  //       $B = load(B)
  //       $C = mul($A, $B)
  //       C = store($C)
  //     }
  //   }
  // }
  // strides:
  // A -> k(c):   1  m(x):  100  n(k):  0
  // B -> k(c): 100  m(x):    0  n(k):  1
  // C -> k(c):   0  m(x):  100  n(k):  1
  // result
  // ------
  // block [] // $matmul
  //     () -> () {
  //   var A : FLOAT32[100:100, 100:1]
  //   var B : FLOAT32[100:100, 100:1]
  //   var C : FLOAT32[100:100, 100:1]
  //   block [] // main
  //       (A, B) -> (C:assign) {
  //     block [m_o:7, n_o:7] // kernel_0
  //         // C[m, n : M, N] = +(A[m, k] * B[k, n])
  //         (A[1600*m_o], B[16*n_o]) -> (C[1600*m_o + 16*n_o]:sum) {
  //       block [k: 100, m_i: 16, n_i: 16]  // VPU2 NCE DPU
  //           16*m_o + m_i < 100
  //           16*n_o + n_i < 100
  //           (A[k_i + 16*m_i], B[k_i + n_i]) -> (C[16*m_i + n_i]) {
  //         $A = load(A)
  //         $B = load(B)
  //         $C = mul($A, $B)
  //         C = store($C)
  //       }
  //     }
  //   }
  // }
}

}  // namespace test
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
