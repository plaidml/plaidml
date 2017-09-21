// Copyright 2017, Vertex.AI. CONFIDENTIAL

#include "tile/hal/opencl/emitocl.h"

#include <map>
#include <utility>

#include "tile/lang/fpconv.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

void Emit::emitType(const sem::Type &t) {
  if (t.region == sem::Type::LOCAL) {
    emit("__local ");
  } else if (t.region == sem::Type::GLOBAL) {
    emit("__global ");
  }
  EmitC::emitType(t);
}

static std::map<std::string, std::string> FuncNameMap = {
    {"recip", "native_recip"}, {"exp", "native_exp"}, {"log", "native_log"}, {"sqrt", "native_sqrt"}};

void Emit::Visit(const sem::CallExpr &n) {
  bool did_override = false;
  auto load = std::dynamic_pointer_cast<sem::LoadExpr>(n.func);
  if (load) {
    auto lookup = std::dynamic_pointer_cast<sem::LookupLVal>(load->inner);
    if (lookup) {
      auto it = FuncNameMap.find(lookup->name);
      if (it != FuncNameMap.end()) {
        emit(it->second);
        did_override = true;
      }
    }
  }
  if (!did_override) {
    n.func->Accept(*this);
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

void Emit::Visit(const sem::IndexExpr &n) {
  switch (n.type) {
    case sem::IndexExpr::GLOBAL:
      emit("get_global_id(" + std::to_string(n.dim) + ")");
      break;
    case sem::IndexExpr::GROUP:
      emit("get_group_id(" + std::to_string(n.dim) + ")");
      break;
    case sem::IndexExpr::LOCAL:
      emit("get_local_id(" + std::to_string(n.dim) + ")");
      break;
    default:
      throw std::runtime_error("Invalid IndexExpr type");
  }
}

void Emit::Visit(const sem::BarrierStmt &n) {
  emitTab();
  emit("barrier(CLK_LOCAL_MEM_FENCE);\n");
}

void Emit::Visit(const sem::Function &n) {
  emit("__kernel ");
  EmitC::Visit(n);
}

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
