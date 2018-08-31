// Copyright 2018, Intel Corp.

#include <gtest/gtest.h>

#include "tile/lang/compose.h"
#include "tile/lang/gen_stripe.h"
#include "tile/lang/intrinsics.h"
#include "tile/lang/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {
namespace test {

lang::RunInfo LoadMatMul() {
  const size_t DIM = 3;
  lang::RunInfo runinfo;
  runinfo.code = "function (A[M, K], B[K, N]) -> (C) { C[m, n : M, N] = +(A[m, k] * B[k, n]); }";
  runinfo.input_shapes.emplace("A", lang::SimpleShape(lang::DataType::FLOAT32, {DIM, DIM}));
  runinfo.input_shapes.emplace("B", lang::SimpleShape(lang::DataType::FLOAT32, {DIM, DIM}));
  runinfo.output_shapes.emplace("C", lang::SimpleShape(lang::DataType::FLOAT32, {DIM, DIM}));
  return runinfo;
}

class VirtualMachine {
 public:
  void ExecuteBlock(const stripe::proto::Block& block, const std::map<std::string, float*>& buffers) {
    std::map<std::string, std::string> agg_ops;
    for (int i = 0; i < block.index_ranges_size(); i++) {
      idxs_.push_back(0);
    }
    for (const auto& ref : block.ref_ins()) {
      auto it = buffers.find(ref.name());
      if (it == buffers.end()) {
        throw std::runtime_error("Missing buffer");
      }
      ptrs_[ref.name()] = it->second;
    }
    for (const auto& ref : block.ref_outs()) {
      agg_ops[ref.name()] = ref.agg_op();
      auto it = buffers.find(ref.name());
      if (it == buffers.end()) {
        throw std::runtime_error("Missing buffer");
      }
      ptrs_[ref.name()] = it->second;
    }
    Loop(block, agg_ops, 0);
  }

 private:
  void Loop(const stripe::proto::Block& block,                  //
            const std::map<std::string, std::string>& agg_ops,  //
            size_t idx) {
    for (size_t i = 0; i < block.index_ranges(idx); i++) {
      LOG(INFO) << "Index Bump: " << block.index_names(idx) << " = " << i;
      idxs_[idx] = i;
      if (idx < block.index_ranges_size() - 1) {
        Loop(block, agg_ops, idx + 1);
      } else {
        ComputeOffsets(block);
        ExecuteStatements(block, agg_ops);
      }
    }
  }

  void ComputeOffsets(const stripe::proto::Block& block) {
    for (const auto& ref : block.ref_ins()) {
      ComputeOffsetFor(block, ref.name(), ref.access());
    }
    for (const auto& ref : block.ref_outs()) {
      ComputeOffsetFor(block, ref.name(), ref.access());
    }
  }

  void ComputeOffsetFor(const stripe::proto::Block& block,  //
                        const std::string& name,            //
                        const stripe::proto::BufferAccess& access) {
    int offset = access.offset();
    std::cout << name << " = ";
    for (int i = 0; i < access.strides_size(); i++) {
      if (i > 0) {
        std::cout << " + ";
      }
      std::cout << access.strides(i) << " * " << block.index_names(i) << "(" << idxs_[i] << ")";
      offset += access.strides(i) * idxs_[i];
    }
    std::cout << " = " << offset << std::endl;
    offsets_[name] = offset;
  }

  void ExecuteStatements(const stripe::proto::Block& block,  //
                         const std::map<std::string, std::string>& agg_ops) {
    for (const auto& stmt : block.stmts()) {
      switch (stmt.op_case()) {
        case stripe::proto::Statement::kLoad: {
          const auto& op = stmt.load();
          vars_[op.into()] = ptrs_[op.from()][offsets_[op.from()]];
        } break;
        case stripe::proto::Statement::kStore: {
          const auto& op = stmt.store();
          auto it = agg_ops.find(op.into());
          if (it->second == lang::intrinsic::SUM) {
            ptrs_[op.into()][offsets_[op.into()]] += vars_[op.from()];
          } else {
            ptrs_[op.into()][offsets_[op.into()]] = vars_[op.from()];
          }
        } break;
        case stripe::proto::Statement::kIntrinsic: {
          const auto& op = stmt.intrinsic();
          if (op.name() == lang::intrinsic::MUL) {
            vars_[op.outputs(0)] = vars_[op.inputs(0)] * vars_[op.inputs(1)];
          }
        } break;
        case stripe::proto::Statement::kConstant:
          break;
        case stripe::proto::Statement::kBlock:
          break;
        default:
          break;
      }
    }
  }

 private:
  std::map<std::string, float*> ptrs_;
  std::map<std::string, int> offsets_;
  std::map<std::string, float> vars_;
  std::vector<int> idxs_;
};

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
  auto main = program.stmts(0).block();

  std::map<std::string, std::vector<float>> data = {
      {"A",
       {
           1, 2, 3,  //
           4, 5, 6,  //
           7, 8, 9   //
       }},
      {"B",
       {
           1, 2, 3,  //
           4, 5, 6,  //
           7, 8, 9   //
       }},
      {"C",
       {
           0, 0, 0,  //
           0, 0, 0,  //
           0, 0, 0   //
       }},
  };

  std::map<std::string, float*> buffers = {
      {"A", data["A"].data()},  //
      {"B", data["B"].data()},
      {"C", data["C"].data()},
  };

  VirtualMachine vm;
  for (const auto& stmt : main.stmts()) {
    if (stmt.has_block()) {
      const auto& kernel = stmt.block();
      vm.ExecuteBlock(kernel, buffers);
    }
  }

  std::cout << "A: " << data["A"] << std::endl;
  std::cout << "B: " << data["B"] << std::endl;
  std::cout << "C: " << data["C"] << std::endl;

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
  //     block [k_o:7, m_o:7, k_o:7] // kernel_0
  //         // C[m, n : M, N] = +(A[m, k] * B[k, n])
  //         (A[16*k_o + 1600*m_o], B[1600*k_o + 16*n_o]) -> (C[1600*m_o + 16*n_o]:sum) {
  //       block [k_i: 16, m_i: 16, n_i: 16]  // VPU2 NCE DPU
  //           16*k_o + k_i < 100
  //           (A[k_i + 16*m_i], B[16*k_i + n_i]) -> (C[16*m_i + n_i]) {
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
