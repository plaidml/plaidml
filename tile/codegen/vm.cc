// Copyright 2018, Intel Corp.

#include "tile/codegen/vm.h"

#include "base/util/printstring.h"
#include "tile/lang/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {
namespace {

using stripe::proto::Intrinsic;

struct NamedIndex {
  std::string name;
  int64_t value;
};

struct Scope {
  std::map<std::string, size_t> offsets;
  std::vector<NamedIndex> global_idxs;
  std::vector<NamedIndex> local_idxs;
};

class VirtualMachine {
 public:
  explicit VirtualMachine(std::map<std::string, std::vector<float>>* buffers)
      : buffers_(*buffers)  //
  {}

  void ExecuteBlock(const stripe::proto::Block& block,  //
                    const Scope& outer,                 //
                    const std::map<std::string, size_t>& offsets) {
    Scope scope;
    scope.offsets = offsets;
    for (int i = 0; i < block.idxs_size(); i++) {
      const auto& idx = block.idxs(i);
      auto name = idx.name();
      scope.local_idxs.emplace_back(NamedIndex{name, 0});
      auto global_idx = 0;
      auto it = std::find_if(outer.local_idxs.begin(), outer.local_idxs.end(),
                             [&name](const NamedIndex& idx) { return idx.name == name; });
      if (it != outer.local_idxs.end()) {
        global_idx = idx.factor() * it->value;
        auto jt = std::find_if(outer.global_idxs.begin(), outer.global_idxs.end(),
                               [&name](const NamedIndex& idx) { return idx.name == name; });
        if (jt != outer.global_idxs.end()) {
          global_idx += jt->value;
        }
      }
      scope.global_idxs.emplace_back(NamedIndex{name, global_idx});
    }
    if (block.idxs_size()) {
      Loop(&scope, block, 0);
    } else {
      ExecuteStatements(&scope, block);
    }
  }

 private:
  void Loop(Scope* scope,                       //
            const stripe::proto::Block& block,  //
            int idx) {
    for (size_t i = 0; i < block.idxs(idx).range(); i++) {
      scope->local_idxs[idx].value = i;
      if (idx < block.idxs_size() - 1) {
        Loop(scope, block, idx + 1);
      } else if (CheckConstraints(*scope, block)) {
        ExecuteStatements(scope, block);
      }
    }
  }

  bool CheckConstraints(const Scope& scope, const stripe::proto::Block& block) {
    for (const auto& constraint : block.constraints()) {
      int lhs = 0;
      for (int i = 0; i < constraint.lhs_size(); i++) {
        lhs += constraint.lhs(i) * scope.global_idxs[i].value + constraint.lhs(i) * scope.local_idxs[i].value;
      }
      if (!(lhs < constraint.rhs())) {
        return false;
      }
    }
    return true;
  }

  size_t ComputeOffsetFor(const Scope& scope,                 //
                          const stripe::proto::Block& block,  //
                          const stripe::proto::BufferAccess& access) {
    int offset = access.offset();
    for (int i = 0; i < access.strides_size(); i++) {
      offset += access.strides(i) * scope.local_idxs[i].value;
    }
    return offset;
  }

  float Load(const std::string& name, size_t offset) {
    auto it = buffers_.find(name);
    if (it == buffers_.end()) {
      throw std::runtime_error("Unknown buffer");
    }
    if (offset >= it->second.size()) {
      throw std::runtime_error(printstring("LOAD: Out of bounds access"));
    }
    return it->second[offset];
  }

  void Store(const std::string& name, size_t offset, float value, const std::string& agg_op) {
    auto it = buffers_.find(name);
    if (it == buffers_.end()) {
      throw std::runtime_error("Unknown buffer");
    }
    if (offset >= it->second.size()) {
      throw std::runtime_error(printstring("STORE: Out of bounds access"));
    }
    if (agg_op == Intrinsic::Value_Name(Intrinsic::SUM)) {
      it->second[offset] += value;
    } else {
      it->second[offset] = value;
    }
  }

  void ExecuteStatements(Scope* scope,                      //
                         const stripe::proto::Block& block  //
  ) {
    std::map<std::string, size_t> offsets;
    std::map<std::string, std::string> agg_ops;
    for (const auto& ref : block.ref_ins()) {
      offsets[ref.into()] = scope->offsets[ref.from()] + ComputeOffsetFor(*scope, block, ref.access());
    }
    for (const auto& ref : block.ref_outs()) {
      offsets[ref.into()] = scope->offsets[ref.from()] + ComputeOffsetFor(*scope, block, ref.access());
      agg_ops[ref.into()] = ref.agg_op();
    }
    std::map<std::string, float> vars;
    for (const auto& stmt : block.stmts()) {
      switch (stmt.op_case()) {
        case stripe::proto::Statement::kLoad: {
          const auto& op = stmt.load();
          vars[op.into()] = Load(op.from(), offsets[op.from()]);
        } break;
        case stripe::proto::Statement::kStore: {
          const auto& op = stmt.store();
          auto it = agg_ops.find(op.into());
          if (it == agg_ops.end()) {
            throw std::runtime_error("Missing agg_op");
          }
          Store(op.into(), offsets[op.into()], vars[op.from()], it->second);
        } break;
        case stripe::proto::Statement::kIntrinsic: {
          const auto& op = stmt.intrinsic();
          if (op.name() == Intrinsic::Value_Name(Intrinsic::MUL)) {
            vars[op.outputs(0)] = vars[op.inputs(0)] * vars[op.inputs(1)];
          }
        } break;
        case stripe::proto::Statement::kConstant:
          break;
        case stripe::proto::Statement::kBlock:
          ExecuteBlock(stmt.block(), *scope, offsets);
          break;
        default:
          break;
      }
    }
  }

 private:
  std::map<std::string, std::vector<float>>& buffers_;
};

}  // namespace

void ExecuteProgram(const stripe::proto::Block& program, std::map<std::string, std::vector<float>>* buffers) {
  Scope scope;
  std::map<std::string, size_t> offsets;
  VirtualMachine vm(buffers);
  vm.ExecuteBlock(program, scope, offsets);
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
