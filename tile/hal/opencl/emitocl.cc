// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/opencl/emitocl.h"

#include <limits>
#include <map>
#include <utility>

#include "base/util/error.h"
#include "tile/lang/exprtype.h"
#include "tile/lang/fpconv.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

static std::map<std::string, std::string> FuncNameMap = {
    {"recip", "native_recip"}, {"exp", "native_exp"}, {"log", "native_log"}, {"sqrt", "native_sqrt"}};

void Emit::Visit(const sem::LoadExpr& n) {
  auto ty = TypeOf(n.inner);
  auto inner = std::dynamic_pointer_cast<sem::SubscriptLVal>(n.inner);
  if (!cl_khr_fp16_ && inner && ty.dtype == DataType::FLOAT16) {
    // Half-width floats can't be loaded directly on this device.
    if (ty.vec_width == 1) {
      emit("vload_half");
    } else {
      emit("vloada_half" + std::to_string(ty.vec_width));
    }
    emit("(");
    inner->offset->Accept(*this);
    emit(", ");
    inner->ptr->Accept(*this);
    emit(")");
  } else {
    EmitC::Visit(n);
  }
}

void Emit::Visit(const sem::StoreStmt& n) {
  auto ty_lhs = TypeOf(n.lhs);
  auto lhs = std::dynamic_pointer_cast<sem::SubscriptLVal>(n.lhs);
  if (!cl_khr_fp16_ && lhs && ty_lhs.dtype == DataType::FLOAT16) {
    // Half-width floats can't be stored directly on this device.
    emitTab();
    if (ty_lhs.vec_width == 1) {
      emit("vstore_half");
    } else {
      emit("vstorea_half" + std::to_string(ty_lhs.vec_width));
    }
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

void Emit::Visit(const sem::DeclareStmt& n) {
  sem::Type ty = n.type;
  sem::Type init_type;
  if (n.init) {
    init_type = TypeOf(n.init);
  }

  if (ty.base == sem::Type::VALUE) {
    if (ty.dtype == DataType::FLOAT16 && !cl_khr_fp16_) {
      ty.dtype = DataType::FLOAT32;
    } else if (ty.dtype == DataType::BOOLEAN) {
      if (n.init) {
        ty.dtype = lang::Promote({init_type}).dtype;
        if (ty.dtype == DataType::BOOLEAN) {
          // If the initializer was booleans, make it INT8.
          ty.dtype = DataType::INT8;
        }
      } else {
        // Assume that this is being initialized from an inter-kernel
        // boolean tensor -- which, in OpenCL, we represent as INT8.
        ty.dtype = DataType::INT8;
      }
    }
  }

  emitTab();
  emitType(ty);
  emit(" ");
  emit(n.name);
  if (n.type.array) {
    emit("[" + std::to_string(n.type.array) + "]");
  }
  if (n.init) {
    emit(" = ");
    if (n.type.array) {
      emit("{");
      for (size_t i = 0; i < n.type.array; i++) {
        n.init->Accept(*this);
        emit(", ");
      }
      emit("}");
    } else {
      EmitWithTypeConversion(init_type, ty, n.init);
    }
  }
  emit(";\n");
  CheckValidType(ty);
  scope_->Bind(n.name, ty);
}

void Emit::Visit(const sem::BinaryExpr& n) {
  auto ty_lhs = TypeOf(n.lhs);
  auto ty_rhs = TypeOf(n.rhs);
  auto ty = lang::Promote({ty_lhs, ty_rhs});
  emit("(");
  EmitWithTypeConversion(ty_lhs, ty, n.lhs);
  emit(" ");
  emit(n.op);
  emit(" ");
  EmitWithTypeConversion(ty_rhs, ty, n.rhs);
  emit(")");
}

void Emit::Visit(const sem::CondExpr& n) {
  auto ty_tcase = TypeOf(n.tcase);
  auto ty_fcase = TypeOf(n.fcase);
  auto ty_cond = TypeOf(n.cond);
  auto ty = lang::Promote({ty_tcase, ty_fcase});
  ty.vec_width = std::max(ty.vec_width, ty_cond.vec_width);
  emit("select(");
  EmitWithTypeConversion(ty_fcase, ty, n.fcase, true);
  emit(", ");
  EmitWithTypeConversion(ty_tcase, ty, n.tcase, true);
  emit(", ");
  EmitWithWidthConversion(ty_cond, ty, n.cond, true);
  emit(")");
}

void Emit::Visit(const sem::SelectExpr& n) {
  auto ty_tcase = TypeOf(n.tcase);
  auto ty_fcase = TypeOf(n.fcase);
  auto ty_cond = TypeOf(n.cond);
  auto ty = lang::Promote({ty_tcase, ty_fcase});
  ty.vec_width = std::max(ty.vec_width, ty_cond.vec_width);
  emit("select(");
  EmitWithTypeConversion(ty_fcase, ty, n.fcase, true);
  emit(", ");
  EmitWithTypeConversion(ty_tcase, ty, n.tcase, true);
  emit(", ");
  EmitWithWidthConversion(ty_cond, ty, n.cond, true);
  emit(")");
}

void Emit::Visit(const sem::ClampExpr& n) {
  auto ty_val = TypeOf(n.val);
  auto ty_min = TypeOf(n.min);
  auto ty_max = TypeOf(n.max);

  // Align value dtypes and vector widths.
  sem::Type ty_clamp{sem::Type::VALUE};
  if (ty_val.base == sem::Type::VALUE) {
    ty_clamp.dtype = ty_val.dtype;
  } else {
    ty_clamp.dtype = DataType::INT32;
  }
  if (ty_min.vec_width != 1) {
    ty_clamp.vec_width = ty_min.vec_width;
  } else {
    ty_clamp.vec_width = ty_max.vec_width;
  }

  emit("clamp(");
  EmitWithTypeConversion(ty_val, ty_clamp, n.val, true);
  emit(", ");
  EmitWithTypeConversion(ty_min, ty_clamp, n.min, true);
  emit(", ");
  EmitWithTypeConversion(ty_max, ty_clamp, n.max, true);
  emit(")");
}

void Emit::Visit(const sem::CastExpr& n) { n.val->Accept(*this); }

void Emit::Visit(const sem::CallExpr& n) {
  auto it = FuncNameMap.find(n.name);
  if (it != FuncNameMap.end()) {
    emit(it->second);
  } else {
    // Assume this is an OpenCL function.
    // TODO: Enumerate the set of callable functions.
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

void Emit::Visit(const sem::IndexExpr& n) {
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

void Emit::Visit(const sem::Block& n) {
  auto previous_scope = scope_;
  lang::Scope<sem::Type> scope{scope_};
  scope_ = &scope;
  EmitC::Visit(n);
  scope_ = previous_scope;
}

void Emit::Visit(const sem::ForStmt& n) {
  auto previous_scope = scope_;
  lang::Scope<sem::Type> scope{scope_};
  scope_ = &scope;
  scope.Bind(n.var, sem::Type{sem::Type::INDEX});
  EmitC::Visit(n);
  scope_ = previous_scope;
}

void Emit::Visit(const sem::BarrierStmt& n) {
  emitTab();
  emit("barrier(CLK_LOCAL_MEM_FENCE);\n");
}

void Emit::Visit(const sem::Function& n) {
  emit("__kernel ");
  lang::Scope<sem::Type> scope;
  scope_ = &scope;

  for (const auto& p : n.params) {
    auto ty = p.first;
    if (ty.dtype == DataType::BOOLEAN) {
      // Global booleans are stored as INT8.
      ty.dtype = DataType::INT8;
    }
    CheckValidType(ty);
    scope.Bind(p.second, ty);
  }

  emitType(n.ret);
  emit(" ");
  emit(n.name);
  emit("(");
  bool first_param = true;
  for (const auto& p : n.params) {
    if (first_param) {
      first_param = false;
    } else {
      emit(", ");
    }
    auto ty = p.first;
    if (!cl_khr_fp16_ && ty.dtype == DataType::FLOAT16) {
      // The device can only use half-width floats as a pointer type, not as a value
      // or a vector element type.  We'll use vloada_half and vstorea_half to access
      // the memory, but the parameter must be declared with a vector width of 1.
      // Note that what's stored in the scope must have the original vector width,
      // which is why we set up the scope using a separate loop.
      ty.vec_width = 1;
    } else if (ty.dtype == DataType::BOOLEAN) {
      // Global booleans are stored as INT8.
      ty.dtype = DataType::INT8;
    }
    emitType(ty);
    emit(" ");
    emit(p.second);
  }
  emit(")\n");
  n.body->Accept(*this);

  scope_ = nullptr;
}

void Emit::CheckValidType(const sem::Type& ty) {
  if (cl_khr_fp64_) {
    return;
  }
  if (ty.base == sem::Type::TVOID || ty.base == sem::Type::INDEX) {
    return;
  }
  if (ty.dtype == DataType::FLOAT64) {
    throw error::Unimplemented{"The device does not support 64-bit floating-point types"};
  }
}

sem::Type Emit::TypeOf(const sem::ExprPtr& expr) { return lang::ExprType::TypeOf(scope_, cl_khr_fp16_, expr); }

sem::Type Emit::TypeOf(const sem::LValPtr& lvalue) { return lang::ExprType::TypeOf(scope_, cl_khr_fp16_, lvalue); }

void Emit::EmitWithTypeConversion(const sem::Type& from, const sem::Type& to, const sem::ExprPtr& expr,
                                  bool force_conversion) {
  if (to.base == sem::Type::POINTER_MUT || to.base == sem::Type::POINTER_CONST) {
    // No conversion required for pointer types.
    expr->Accept(*this);
    return;
  }
  if (!force_conversion && ((from.vec_width == 1 && from.base == sem::Type::VALUE && is_int(from.dtype) &&
                             (to.base == sem::Type::INDEX || (to.base == sem::Type::VALUE && is_int(to.dtype)))) ||
                            (from.base == to.base && from.dtype == to.dtype && from.vec_width == to.vec_width))) {
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

void Emit::EmitWithWidthConversion(const sem::Type& from, const sem::Type& to, const sem::ExprPtr& expr,
                                   bool force_conversion) {
  if (to.base == sem::Type::POINTER_MUT || to.base == sem::Type::POINTER_CONST) {
    // No conversion required.
    expr->Accept(*this);
    return;
  }

  // Based on the target type, convert to the appropriate condition type.
  sem::Type condition_type = to;
  switch (condition_type.dtype) {
    case DataType::BOOLEAN:
    case DataType::INT8:
    case DataType::UINT8:
      condition_type.dtype = DataType::INT8;
      break;
    case DataType::INT16:
    case DataType::UINT16:
    case DataType::FLOAT16:
      condition_type.dtype = DataType::INT16;
      break;
    case DataType::INT32:
    case DataType::UINT32:
    case DataType::FLOAT32:
      condition_type.dtype = DataType::INT32;
      break;
    case DataType::INT64:
    case DataType::UINT64:
    case DataType::FLOAT64:
      condition_type.dtype = DataType::INT64;
      break;
    default:
      break;
  }

  EmitWithTypeConversion(from, condition_type, expr, force_conversion);
  if (from.vec_width != to.vec_width) {
    // We need to convert a scalar into a vector.
    // See the OpenCL 1.2 spec (pg 219, Section 6.3: Operators, e.)
    emit(" != ");
    emit("(");
    EmitC::emitType(condition_type);
    emit(")");
    emit("0");
  }
}

void Emit::emitType(const sem::Type& t) {
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
