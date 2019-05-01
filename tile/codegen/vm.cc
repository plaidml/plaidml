// Copyright 2018, Intel Corporation

#include "tile/codegen/vm.h"

#include <algorithm>

#include <boost/format.hpp>

#include "base/util/lookup.h"
#include "base/util/stream_container.h"
#include "base/util/throw.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

using namespace stripe;  // NOLINT

namespace {

std::map<std::string, std::function<float(float, float)>> BINARY_OPS = {
    {"add", [](float a, float b) { return a + b; }},
    {"mul", [](float a, float b) { return a * b; }},
    {"cmp_lt", [](float a, float b) { return a < b; }},
};

std::map<std::string, std::function<float(float, float, float)>> TERNARY_OPS = {
    {"cond", [](float c, float t, float f) { return c ? t : f; }},
};

class Scope {
 public:
  Scope() {}
  explicit Scope(Scope* outer) : outer_(outer), depth_(outer->depth_ + 1) {}

  void ExecuteProgram(const Block& block, std::map<std::string, Buffer>* buffers) {
    Scope outer;
    outer_ = &outer;
    std::map<std::string, Buffer> tmps;
    for (const auto& ref : block.refs) {
      assert(ref.from.empty());
      if (ref.has_tag("user")) {
        refs_[ref.into()] = &safe_at(buffers, ref.into());
      } else {
        Buffer buf(ref.interior_shape.elem_size());
        tmps.emplace(ref.into(), buf);
        refs_[ref.into()] = &safe_at(&tmps, ref.into());
      }
    }
    ExecuteStatements(block);
  }

 private:
  void ExecuteBlock(const Block& block) {
    IVLOG(4, Tab() << "ExecuteBlock: " << block.name);
    std::map<std::string, Buffer> buffers;
    for (const auto& ref : block.refs) {
      if (ref.from.empty()) {
        Buffer buf(ref.interior_shape.elem_size());
        buffers.emplace(ref.into(), buf);
        refs_[ref.into()] = &safe_at(&buffers, ref.into());
      } else {
        refs_[ref.into()] = safe_at(outer_->refs_, ref.from);
      }
    }
    if (block.idxs.size()) {
      Loop(block, 0);
    } else {
      ExecuteStatements(block);
    }
  }

  void Loop(const Block& block, size_t depth) {
    const auto& idx = block.idxs[depth];
    auto base = idx.affine.eval(outer_->idxs_);
    for (size_t i = 0; i < idx.range; i++) {
      idxs_[idx.name] = base + i;
      if (depth < block.idxs.size() - 1) {
        Loop(block, depth + 1);
      } else {
        ExecuteStatements(block);
      }
    }
  }

  bool CheckConstraints(const Block& block) {
    for (const auto& constraint : block.constraints) {
      if (constraint.eval(idxs_) < 0) {
        return false;
      }
    }
    return true;
  }

  size_t ComputeOffsetFor(const Block& block, const Refinement& ref) {
    int offset = 0;
    if (!ref.from.empty()) {
      offset = safe_at(outer_->offsets_, ref.from);
    }
    std::stringstream ss;
    ss << "ref: " << ref.into() << ", offset = " << offset;
    assert(ref.interior_shape.dims.size() == ref.access.size());
    for (size_t i = 0; i < ref.interior_shape.dims.size(); i++) {
      auto access = ref.access[i].eval(idxs_);
      auto stride = ref.interior_shape.dims[i].stride;
      offset += access * stride;
      ss << " + (" << access << " * " << stride << ")";
    }
    ss << " = " << offset;
    IVLOG(5, Tab() << ss.str());
    return offset;
  }

  float DoLoad(const std::string& name, size_t offset) {
    auto it = refs_.find(name);
    if (it == refs_.end()) {
      throw_with_trace(std::runtime_error("Unknown buffer"));
    }
    if (offset >= it->second->size()) {
      throw_with_trace(
          std::runtime_error(str(boost::format("LOAD: Out of bounds access on '%s', offset: %zu, size: %zu") %  //
                                 name % offset % it->second->size())));
    }
    return (*it->second)[offset];
  }

  void DoStore(const std::string& name, size_t offset, float value, const std::string& agg_op) {
    auto it = refs_.find(name);
    if (it == refs_.end()) {
      throw_with_trace(std::runtime_error("Unknown buffer"));
    }
    if (offset >= it->second->size()) {
      throw_with_trace(
          std::runtime_error(str(boost::format("STORE: Out of bounds access on '%s', offset: %zu, size: %zu") %  //
                                 name % offset % it->second->size())));
    }
    if (agg_op == Intrinsic::SUM) {
      (*it->second)[offset] += value;
    } else {
      (*it->second)[offset] = value;
    }
  }

  float DoLoadIndex(const Affine& idx) { return idx.eval(idxs_); }

  void ExecuteStatements(const Block& block) {
    if (!CheckConstraints(block)) {
      return;
    }
    std::map<std::string, float> vars;
    for (const auto& ref : block.refs) {
      offsets_[ref.into()] = ComputeOffsetFor(block, ref);
    }
    IVLOG(5, Tab() << "idxs: " << StreamContainer(idxs_));
    IVLOG(5, Tab() << "offsets: " << StreamContainer(offsets_));
    for (const auto& stmt : block.stmts) {
      switch (stmt->kind()) {
        case StmtKind::Load: {
          const auto& op = Load::Downcast(stmt);
          vars[op->into] = DoLoad(op->from, offsets_[op->from]);
        } break;
        case StmtKind::Store: {
          const auto& op = Store::Downcast(stmt);
          auto it = block.ref_by_into(op->into, false);
          if (it == block.refs.end()) {
            throw_with_trace(std::runtime_error("Missing agg_op"));
          }
          DoStore(op->into, offsets_[op->into], vars[op->from], it->agg_op);
        } break;
        case StmtKind::LoadIndex: {
          const auto& op = LoadIndex::Downcast(stmt);
          vars[op->into] = DoLoadIndex(op->from);
        } break;
        case StmtKind::Intrinsic: {
          const auto& op = Intrinsic::Downcast(stmt);
          switch (op->inputs.size()) {
            case 2: {
              auto it = BINARY_OPS.find(op->name);
              if (it == BINARY_OPS.end()) {
                throw_with_trace(std::runtime_error(str(boost::format("Unsupported binary intrinsic: %s") % op->name)));
              }
              vars[op->outputs[0]] = it->second(vars[op->inputs[0]], vars[op->inputs[1]]);
            } break;
            case 3: {
              auto it = TERNARY_OPS.find(op->name);
              if (it == TERNARY_OPS.end()) {
                throw_with_trace(
                    std::runtime_error(str(boost::format("Unsupported ternary intrinsic: %s") % op->name)));
              }
              vars[op->outputs[0]] = it->second(vars[op->inputs[0]], vars[op->inputs[1]], vars[op->inputs[2]]);
            } break;
            default:
              throw_with_trace(std::runtime_error(
                  str(boost::format("Unsupported number of operands for intrinsic: %s") % op->name)));
              break;
          }
        } break;
        case StmtKind::Constant: {
          const auto& op = Constant::Downcast(stmt);
          switch (op->type) {
            case ConstType::Integer:
              vars[op->name] = op->iconst;
              break;
            case ConstType::Float:
              vars[op->name] = op->fconst;
              break;
          }
        } break;
        case StmtKind::Block: {
          Scope scope(this);
          scope.ExecuteBlock(*Block::Downcast(stmt));
        } break;
        default:
          break;
      }
    }
  }

  std::string Tab() const { return std::string(depth_ * 2, ' '); }

 private:
  Scope* outer_ = nullptr;
  size_t depth_ = 0;
  std::map<std::string, int64_t> idxs_;
  std::map<std::string, Buffer*> refs_;
  std::map<std::string, size_t> offsets_;
};

}  // namespace

void ExecuteProgram(const Block& program, std::map<std::string, Buffer>* buffers) {
  Scope scope;
  scope.ExecuteProgram(program, buffers);
}

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
