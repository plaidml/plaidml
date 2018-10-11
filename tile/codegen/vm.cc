// Copyright 2018, Intel Corp.

#include "tile/codegen/vm.h"

#include <algorithm>

#include "base/util/printstring.h"
#include "base/util/stream_container.h"
#include "tile/stripe/stripe.h"

namespace std {

ostream& operator<<(ostream& os, const pair<string, int64_t>& item) {
  os << item.first << ":" << item.second;
  return os;
}

}  // namespace std

namespace vertexai {
namespace tile {
namespace codegen {
namespace {

using namespace stripe;  // NOLINT

struct Scope {
  std::map<std::string, int64_t> global_idxs;
  std::map<std::string, int64_t> idxs;
};

class VirtualMachine {
 public:
  explicit VirtualMachine(std::map<std::string, std::vector<float>>* buffers)
      : buffers_(*buffers)  //
  {}

  void ExecuteBlock(const Block& block, const Scope& outer) {
    Scope scope;
    for (size_t i = 0; i < block.idxs.size(); i++) {
      const auto& idx = block.idxs[i];
      auto global_idx = 0;
      auto it = outer.idxs.find(idx.from);
      if (it != outer.idxs.end()) {
        global_idx = idx.factor * it->second;
      }
      scope.global_idxs.insert(std::make_pair(idx.name, global_idx));
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
      auto idx_name = block.idxs[idx].name;
      scope->idxs[idx_name] = scope->global_idxs[idx_name] + i;
      if (idx < block.idxs.size() - 1) {
        Loop(scope, block, idx + 1);
      } else if (CheckConstraints(*scope, block)) {
        ExecuteStatements(scope, block);
      }
    }
  }

  bool CheckConstraints(const Scope& scope, const Block& block) {
    for (const auto& constraint : block.constraints) {
      auto result = constraint.eval(scope.idxs);
      if (result < 0) {
        return false;
      }
    }
    return true;
  }

  size_t ComputeOffsetFor(const Scope& scope, const Block& block, const Refinement& ref) {
    int offset = 0;
    std::stringstream ss;
    ss << "ref: " << ref.into << ", offset = ";
    assert(ref.shape.dims.size() == ref.access.size());
    for (size_t i = 0; i < ref.shape.dims.size(); i++) {
      auto access = ref.access[i].eval(scope.idxs);
      auto stride = ref.shape.dims[i].stride;
      offset += access * stride;
      if (i > 0) {
        ss << " + ";
      }
      ss << "(" << access << " * " << stride << ")";
    }
    ss << " = " << offset;
    IVLOG(5, ss.str());
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
    std::map<std::string, float> vars;
    std::map<std::string, size_t> offsets;
    IVLOG(5, "idxs: " << StreamContainer(scope->idxs));
    for (const auto& ref : block.refs) {
      offsets[ref.into] = ComputeOffsetFor(*scope, block, ref);
    }
    for (const auto& stmt : block.stmts) {
      switch (stmt->kind()) {
        case StmtKind::Load: {
          const auto& op = Load::Downcast(stmt);
          vars[op->into] = DoLoad(op->from, offsets[op->from]);
        } break;
        case StmtKind::Store: {
          const auto& op = Store::Downcast(stmt);
          auto it = block.ref_by_into(op->into);
          if (it == block.refs.end()) {
            throw std::runtime_error("Missing agg_op");
          }
          DoStore(op->into, offsets[op->into], vars[op->from], it->agg_op);
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
          ExecuteBlock(*Block::Downcast(stmt), *scope);
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
  VirtualMachine vm(buffers);
  vm.ExecuteBlock(program, scope);
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
