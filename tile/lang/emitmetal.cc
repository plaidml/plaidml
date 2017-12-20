
#include "tile/lang/emitmetal.h"

#include <map>
#include <utility>

namespace vertexai {
namespace tile {
namespace lang {

void EmitMetal::emitType(const sem::Type& t) {
  if (t.region == sem::Type::LOCAL) {
    emit("threadgroup ");
  } else if (t.region == sem::Type::GLOBAL) {
    emit("device ");
  }
  EmitC::emitType(t);
}

static std::map<std::string, std::string> FuncNameMap = {
    {"recip", "native_recip"}, {"exp", "native_exp"}, {"log", "native_log"}, {"sqrt", "native_sqrt"},
};

void EmitMetal::Visit(const sem::CallExpr& n) {
  auto it = FuncNameMap.find(n.name);
  if (it != FuncNameMap.end()) {
    emit(it->second);
  } else {
    emit(n.name);
  }
  emit("(");
  for (size_t i = 0; i < n.vals.size(); i++) {
    n.vals[i]->Accept(*this);
    if (i != n.vals.size() - 1) {
      emit(", ");
    }
  }
  emit(")");
}

void EmitMetal::Visit(const sem::IndexExpr& n) {
  switch (n.type) {
    case sem::IndexExpr::GLOBAL:
      emit("_globalid[" + std::to_string(n.dim) + "]");
      break;
    case sem::IndexExpr::GROUP:
      emit("_groupid[" + std::to_string(n.dim) + "]");
      break;
    case sem::IndexExpr::LOCAL:
      emit("_tid");
      break;
    default:
      throw std::runtime_error("Invalid IndexExpr type");
  }
}

void EmitMetal::Visit(const sem::BarrierStmt& n) {
  emitTab();
  emit("threadgroup_barrier(mem_flags::mem_threadgroup);\n");
}

void EmitMetal::Visit(const sem::Function& n) {
  emit("kernel ");
  emitType(n.ret);
  emit(" ");
  emit(n.name);
  emit("(\n");
  for (size_t i = 0; i < n.params.size(); i++) {
    const auto& p = n.params[i];
    emit("    ");
    emitType(p.first);
    emit(" ");
    emit(p.second);
    emit(" [[ buffer(" + std::to_string(i) + ") ]],\n");
  }
  emit("    uint _tid [[ thread_index_in_threadgroup ]],\n");
  emit("    uint3 _groupid [[ threadgroup_position_in_grid ]],\n");
  emit("    uint3 _globalid [[ thread_position_in_grid ]]\n");
  emit(")\n");
  n.body->Accept(*this);
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
