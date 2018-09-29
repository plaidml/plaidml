// Copyright 2018, Intel Corp.

#include "tile/codegen/vm.h"

#include <algorithm>

#include "base/util/printstring.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {
namespace {

using namespace stripe;  // NOLINT

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

  void ExecuteBlock(const Block& block, const Scope& outer, const std::map<std::string, size_t>& offsets) {
    Scope scope;
    scope.offsets = offsets;
    for (size_t i = 0; i < block.idxs.size(); i++) {
      const auto& idx = block.idxs[i];
      auto name = idx.name;
      scope.local_idxs.emplace_back(NamedIndex{name, 0});
      auto global_idx = 0;
      auto it = std::find_if(outer.local_idxs.begin(), outer.local_idxs.end(),
                             [&name](const NamedIndex& idx) { return idx.name == name; });
      if (it != outer.local_idxs.end()) {
        global_idx = idx.factor * it->value;
        auto jt = std::find_if(outer.global_idxs.begin(), outer.global_idxs.end(),
                               [&name](const NamedIndex& idx) { return idx.name == name; });
        if (jt != outer.global_idxs.end()) {
          global_idx += jt->value;
        }
      }
      scope.global_idxs.emplace_back(NamedIndex{name, global_idx});
    }
    if (block.idxs.size()) {
      Loop(&scope, block, 0);
    } else {
      ExecuteStatements(&scope, block);
    }
  }

 private:
  void Loop(Scope* scope, const Block& block, size_t idx) {
    for (size_t i = 0; i < block.idxs[idx].range; i++) {
      scope->local_idxs[idx].value = i;
      if (idx < block.idxs.size() - 1) {
        Loop(scope, block, idx + 1);
      } else if (CheckConstraints(*scope, block)) {
        ExecuteStatements(scope, block);
      }
    }
  }

  bool CheckConstraints(const Scope& scope, const Block& block) {
    for (const auto& constraint : block.constraints) {
      int lhs = 0;
      for (size_t i = 0; i < constraint.lhs.size(); i++) {
        lhs += constraint.lhs[i] * scope.global_idxs[i].value + constraint.lhs[i] * scope.local_idxs[i].value;
      }
      if (!(lhs < constraint.rhs)) {
        return false;
      }
    }
    return true;
  }

  size_t ComputeOffsetFor(const Scope& scope, const Block& block, const BufferAccess& access) {
    int offset = access.offset;
    for (size_t i = 0; i < access.strides.size(); i++) {
      offset += access.strides[i] * scope.local_idxs[i].value;
    }
    return offset;
  }

  float DoLoad(const std::string& name, size_t offset) {
    auto it = buffers_.find(name);
    if (it == buffers_.end()) {
      throw std::runtime_error("Unknown buffer");
    }
    if (offset >= it->second.size()) {
      throw std::runtime_error(printstring("LOAD: Out of bounds access"));
    }
    return it->second[offset];
  }

  void DoStore(const std::string& name, size_t offset, float value, const std::string& agg_op) {
    auto it = buffers_.find(name);
    if (it == buffers_.end()) {
      throw std::runtime_error("Unknown buffer");
    }
    if (offset >= it->second.size()) {
      throw std::runtime_error(printstring("STORE: Out of bounds access"));
    }
    if (agg_op == Intrinsic::SUM) {
      it->second[offset] += value;
    } else {
      it->second[offset] = value;
    }
  }

  void ExecuteStatements(Scope* scope, const Block& block) {
    std::map<std::string, size_t> offsets;
    std::map<std::string, const Refinement*> refs_by_into;
    for (const auto& ref : block.refs) {
      offsets[ref.into] = scope->offsets[ref.from] + ComputeOffsetFor(*scope, block, ref.access);
      refs_by_into.insert(std::make_pair(ref.into, &ref));
    }
    std::map<std::string, float> vars;
    for (const auto& stmt : block.stmts) {
      switch (stmt->kind()) {
        case StmtKind::Load: {
          const auto& op = Load::Downcast(stmt);
          vars[op->into] = DoLoad(op->from, offsets[op->from]);
        } break;
        case StmtKind::Store: {
          const auto& op = Store::Downcast(stmt);
          auto it = refs_by_into.find(op->into);
          if (it == refs_by_into.end()) {
            throw std::runtime_error("Missing agg_op");
          }
          DoStore(op->into, offsets[op->into], vars[op->from], it->second->agg_op);
        } break;
        case StmtKind::Intrinsic: {
          const auto& op = Intrinsic::Downcast(stmt);
          if (op->name == Intrinsic::MUL) {
            vars[op->outputs[0]] = vars[op->inputs[0]] * vars[op->inputs[1]];
          }
        } break;
        case StmtKind::Constant:
          break;
        case StmtKind::Block:
          ExecuteBlock(*Block::Downcast(stmt), *scope, offsets);
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

void ExecuteProgram(const Block& program, std::map<std::string, std::vector<float>>* buffers) {
  Scope scope;
  std::map<std::string, size_t> offsets;
  VirtualMachine vm(buffers);
  vm.ExecuteBlock(program, scope, offsets);
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
