// Copyright 2018, Intel Corp.

#include "tile/codegen/vm.h"

#include "base/util/logging.h"
#include "tile/lang/intrinsics.h"
#include "tile/lang/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {
namespace {

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
            int idx) {
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

}  // namespace

void ExecuteProgram(const stripe::proto::Block& program, const std::map<std::string, float*>& buffers) {
  VirtualMachine vm;
  auto main = program.stmts(0).block();
  for (const auto& stmt : main.stmts()) {
    if (stmt.has_block()) {
      const auto& kernel = stmt.block();
      vm.ExecuteBlock(kernel, buffers);
    }
  }
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
