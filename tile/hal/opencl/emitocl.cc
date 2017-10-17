// Copyright 2017, Vertex.AI.

#include "tile/hal/opencl/emitocl.h"

#include <limits>
#include <map>
#include <utility>

#include "tile/hal/opencl/exprtype.h"
#include "tile/lang/fpconv.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

static std::map<std::string, std::string> FuncNameMap = {
    {"recip", "native_recip"}, {"exp", "native_exp"}, {"log", "native_log"}, {"sqrt", "native_sqrt"}};

void Emit::Visit(const sem::LoadExpr &n) {
  auto ty = TypeOf(n.inner);
  auto inner = std::dynamic_pointer_cast<sem::SubscriptLVal>(n.inner);
  if (!cl_khr_fp16_ && inner && ty.dtype == lang::DataType::FLOAT16) {
    // Half-width floats can't be loaded directly on this device.
    std::string fname = "vloada_half";
    if (ty.vec_width != 1) {
      fname = fname + std::to_string(ty.vec_width);
    }
    emit(fname);
    emit("(");
    inner->offset->Accept(*this);
    emit(", ");
    inner->ptr->Accept(*this);
    emit(")");
  } else {
    EmitC::Visit(n);
  }
}

void Emit::Visit(const sem::StoreStmt &n) {
  auto ty_lhs = TypeOf(n.lhs);
  auto lhs = std::dynamic_pointer_cast<sem::SubscriptLVal>(n.lhs);
  if (!cl_khr_fp16_ && lhs && ty_lhs.dtype == lang::DataType::FLOAT16) {
    // Half-width floats can't be stored directly on this device.
    std::string fname = "vstorea_half";
    if (ty_lhs.vec_width != 1) {
      fname = fname + std::to_string(ty_lhs.vec_width);
    }
    emitTab();
    emit(fname);
    emit("(");
    n.rhs->Accept(*this);
    emit(", ");
    lhs->offset->Accept(*this);
    emit(", ");
    lhs->ptr->Accept(*this);
    emit(");\n");
  } else {
    emitTab();
    n.lhs->Accept(*this);
    emit(" = ");
    EmitWithTypeConversion(TypeOf(n.rhs), ty_lhs, n.rhs);
    emit(";\n");
  }
}

void Emit::Visit(const sem::DeclareStmt &n) {
  sem::DeclareStmt decl(n);
  if (decl.type.base == sem::Type::VALUE && ((decl.type.dtype == lang::DataType::FLOAT16 && !cl_khr_fp16_) ||
                                             (decl.type.dtype == lang::DataType::BOOLEAN && 1 < decl.type.vec_width))) {
    sem::Type ty = decl.type;
    if (decl.init) {
      IVLOG(5, "Determining type of initializer for " << decl.name);
      ty = Promote({ty, TypeOf(decl.init)});
    }
    if (ty.dtype == lang::DataType::FLOAT16 && !cl_khr_fp16_) {
      ty.dtype = lang::DataType::FLOAT32;
    } else if (ty.dtype == lang::DataType::BOOLEAN && 1 < decl.type.vec_width) {
      ty.dtype = lang::DataType::INT8;
    }
    decl.type.dtype = ty.dtype;
    IVLOG(5, "... type is: " << decl.type);
  }
  EmitC::Visit(decl);
  scope_->Bind(decl.name, decl.type);
}

void Emit::Visit(const sem::BinaryExpr &n) {
  auto ty_lhs = TypeOf(n.lhs);
  auto ty_rhs = TypeOf(n.rhs);
  auto ty = Promote({ty_lhs, ty_rhs});
  emit("(");
  EmitWithTypeConversion(ty_lhs, ty, n.lhs);
  emit(n.op);
  EmitWithTypeConversion(ty_rhs, ty, n.rhs);
  emit(")");
}

void Emit::Visit(const sem::SelectExpr &n) {
  auto ty_tcase = TypeOf(n.tcase);
  auto ty_fcase = TypeOf(n.fcase);
  auto ty = Promote({ty_tcase, ty_fcase});
  emit("select(");
  EmitWithTypeConversion(ty_fcase, ty, n.fcase);
  emit(", ");
  EmitWithTypeConversion(ty_tcase, ty, n.tcase);
  emit(", ");
  EmitWithWidthConversion(TypeOf(n.cond), ty, n.cond);
  emit(")");
}

void Emit::Visit(const sem::ClampExpr &n) {
  auto ty_val = TypeOf(n.val);
  auto ty_min = TypeOf(n.min);
  auto ty_max = TypeOf(n.max);

  // Align value dtypes and vector widths.
  sem::Type ty_clamp{sem::Type::VALUE};
  if (ty_val.base == sem::Type::VALUE) {
    ty_clamp.dtype = ty_val.dtype;
  } else {
    ty_clamp.dtype = lang::DataType::INT32;
  }
  if (ty_min.vec_width != 1) {
    ty_clamp.vec_width = ty_min.vec_width;
  } else {
    ty_clamp.vec_width = ty_max.vec_width;
  }

  emit("clamp(");
  n.val->Accept(*this);
  emit(", ");
  EmitWithTypeConversion(ty_min, ty_clamp, n.min);
  emit(", ");
  EmitWithTypeConversion(ty_max, ty_clamp, n.max);
  emit(")");
}

void Emit::Visit(const sem::CastExpr &n) { n.val->Accept(*this); }

void Emit::Visit(const sem::CallExpr &n) {
  bool did_override = false;
  auto load = std::dynamic_pointer_cast<sem::LoadExpr>(n.func);
  if (load) {
    auto lookup = std::dynamic_pointer_cast<sem::LookupLVal>(load->inner);
    if (lookup) {
      auto it = FuncNameMap.find(lookup->name);
      if (it != FuncNameMap.end()) {
        emit(it->second);
      } else {
        // Assume this is an OpenCL function.
        // TODO: Enumerate the set of callable functions.
        emit(lookup->name);
      }
      did_override = true;
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

void Emit::Visit(const sem::Block &n) {
  auto previous_scope = scope_;
  lang::Scope<sem::Type> scope{scope_};
  scope_ = &scope;
  EmitC::Visit(n);
  scope_ = previous_scope;
}

void Emit::Visit(const sem::ForStmt &n) {
  auto previous_scope = scope_;
  lang::Scope<sem::Type> scope{scope_};
  scope_ = &scope;
  scope.Bind(n.var, sem::Type{sem::Type::INDEX});
  EmitC::Visit(n);
  scope_ = previous_scope;
}

void Emit::Visit(const sem::BarrierStmt &n) {
  emitTab();
  emit("barrier(CLK_LOCAL_MEM_FENCE);\n");
}

void Emit::Visit(const sem::Function &n) {
  emit("__kernel ");
  lang::Scope<sem::Type> scope;
  scope_ = &scope;

  for (const auto &p : n.params) {
    auto ty = p.first;
    if (ty.dtype == lang::DataType::BOOLEAN) {
      // Global booleans are stored as INT8.
      ty.dtype = lang::DataType::INT8;
    }
    scope.Bind(p.second, ty);
  }

  emitType(n.ret);
  emit(" ");
  emit(n.name);
  emit("(");
  bool first_param = true;
  for (const auto &p : n.params) {
    if (first_param) {
      first_param = false;
    } else {
      emit(", ");
    }
    auto ty = p.first;
    if (!cl_khr_fp16_ && ty.dtype == lang::DataType::FLOAT16) {
      // The device can only use half-width floats as a pointer type, not as a value
      // or a vector element type.  We'll use vloada_half and vstorea_half to access
      // the memory, but the parameter must be declared with a vector width of 1.
      // Note that what's stored in the scope must have the original vector width,
      // which is why we set up the scope using a separate loop.
      ty.vec_width = 1;
    } else if (ty.dtype == lang::DataType::BOOLEAN) {
      // Global booleans are stored as INT8.
      ty.dtype = lang::DataType::INT8;
    }
    emitType(ty);
    emit(" ");
    emit(p.second);
  }
  emit(")\n");
  n.body->Accept(*this);

  scope_ = nullptr;
}

sem::Type Emit::TypeOf(const sem::ExprPtr &expr) { return ExprType::TypeOf(scope_, cl_khr_fp16_, expr); }

sem::Type Emit::TypeOf(const sem::LValPtr &lvalue) { return ExprType::TypeOf(scope_, cl_khr_fp16_, lvalue); }

void Emit::EmitWithTypeConversion(const sem::Type &from, const sem::Type &to, const sem::ExprPtr &expr) {
  if (to.base == sem::Type::POINTER_MUT || to.base == sem::Type::POINTER_CONST ||
      (from.vec_width == 1 && from.base == sem::Type::VALUE &&
       (from.dtype == lang::DataType::INT32 || from.dtype == lang::DataType::INT16 ||
        from.dtype == lang::DataType::INT8) &&
       (to.base == sem::Type::INDEX ||
        (to.base == sem::Type::VALUE && (to.dtype == lang::DataType::INT32 || to.dtype == lang::DataType::INT16 ||
                                         to.dtype == lang::DataType::INT8)))) ||
      (from.base == to.base && from.dtype == to.dtype && from.vec_width == to.vec_width)) {
    // No conversion required.
    expr->Accept(*this);
    return;
  }
  if (from.base == sem::Type::INDEX || (from.base == sem::Type::VALUE && from.vec_width == 1)) {
    emit("(");
    EmitC::emitType(to);
    emit(")");
    expr->Accept(*this);
    return;
  }
  emit("convert_");
  EmitC::emitType(to);
  emit("(");
  expr->Accept(*this);
  emit(")");
}

void Emit::EmitWithWidthConversion(const sem::Type &from, const sem::Type &to, const sem::ExprPtr &expr) {
  if (to.base == sem::Type::POINTER_MUT || to.base == sem::Type::POINTER_CONST) {
    // No conversion required.
    expr->Accept(*this);
    return;
  }
  if (from.base == sem::Type::VALUE && from.vec_width == to.vec_width) {
    expr->Accept(*this);
    return;
  }
  sem::Type etype = from;
  etype.base = sem::Type::VALUE;
  etype.vec_width = to.vec_width;
  if (from.vec_width == 1) {
    // We can (and should) convert it with a simple cast.
    emit("(");
    EmitC::emitType(etype);
    emit(")");
    expr->Accept(*this);
    return;
  }
  emit("convert_");
  EmitC::emitType(etype);
  emit("((");
  EmitC::emitType(from);
  emit(")");
  expr->Accept(*this);
  emit(")");
}

void Emit::emitType(const sem::Type &t) {
  if (t.region == sem::Type::LOCAL) {
    emit("__local ");
  } else if (t.region == sem::Type::GLOBAL) {
    emit("__global ");
  }
  EmitC::emitType(t);
}

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
