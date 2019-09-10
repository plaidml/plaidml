// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/emitcm.h"

#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "base/util/error.h"
#include "tile/lang/exprtype.h"
#include "tile/lang/fpconv.h"

#include "base/util/env.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

void Emit::Visit(const sem::LoadExpr& n) {
  auto ty = TypeOf(n.inner);
  auto s = GetGlobalVarWithOffset(n.inner);
  if (s.length() > 0 && input_replace_map.find(s) != input_replace_map.end()) {
    emit(input_replace_map[s]);
    return;
  }
  auto inner = std::dynamic_pointer_cast<sem::SubscriptLVal>(n.inner);
  if (inner && GetGlobalVarWithOffset(inner).size() > 0) {
    if (!single_element_rw_mode) {
      int stride = GetIndexStride(inner->offset);
      if (stride > 1) {
        inner->ptr->Accept(*this);
        emit(", ");
        inner->offset->Accept(*this);
        emit(", ");
        emit("element_offset_");
        emit(std::to_string(stride));
        return;
      }
    }
    inner->ptr->Accept(*this);
    emit(", sizeof(");
    emitType(ty);
    emit(") * ");
    inner->offset->Accept(*this);
    if (IsVector(inner->offset)) {
      emit("(0)");
    }
  } else {
    n.inner->Accept(*this);
  }
}

void Emit::Visit(const sem::StoreStmt& n) {
  auto ty_lhs = TypeOf(n.lhs);
  auto s = GetGlobalVarWithOffset(n.lhs);
  auto rhs_load_exp = std::dynamic_pointer_cast<sem::LoadExpr>(n.rhs);

  if (s.size() > 0) {
    AssignGlobalVarToTemp(n.rhs);

    auto int_const = std::dynamic_pointer_cast<sem::IntConst>(n.rhs);
    if (single_element_rw_mode || int_const) {
      EmitSingleElementWriteStat(n.lhs, n.rhs);
      return;
    }
    if (rhs_load_exp && GetGlobalVarWithOffset(n.rhs).size() > 0) {
      EmitWriteStat(n.lhs, n.rhs);
    } else {
      auto cond_expr = std::dynamic_pointer_cast<sem::CondExpr>(n.rhs);
      if (cond_expr) {
        std::string temp = "cm_temp" + std::to_string(temp_num);
        temp_num++;
        EmitVector(ty_lhs, vector_size, temp);
        emit(";\n");

        emitTab();
        emit(temp);
        emit(".");
        n.rhs->Accept(*this);
        emit(";\n");

        EmitWriteStat(n.lhs, temp);
        return;
      }
      EmitWriteStat(n.lhs, n.rhs);
    }
  } else {
    if (rhs_load_exp && GetGlobalVarWithOffset(n.rhs).size() > 0) {
      EmitReadStat(n.lhs, n.rhs);
      vector_stride_map[GetLValueName(n.lhs)] = GetIndexStride(n.rhs);
    } else {
      emitTab();
      n.lhs->Accept(*this);

      auto cond_exp = std::dynamic_pointer_cast<sem::CondExpr>(n.rhs);
      auto select_exp = std::dynamic_pointer_cast<sem::SelectExpr>(n.rhs);
      std::string op = (cond_exp || select_exp) ? "." : " = ";

      emit(op);
      n.rhs->Accept(*this);
      emit(";\n");
    }
  }
}

void Emit::Visit(const sem::DeclareStmt& n) {
  sem::Type ty = n.type;
  if (ty.dtype == DataType::BOOLEAN) {
    ty.dtype = DataType::INT8;
  }

  if (n.init) {
    if (ty.base == sem::Type::INDEX) {
      int stride = GetIndexStride(n.init);
      tv.index_stride_map[n.name] = stride;

      if (stride > 1) {
        std::string vname = "element_offset_" + std::to_string(stride);
        if (tv.vector_params.find(vname) == tv.vector_params.end()) {
          if (!single_element_rw_mode) {
            emitTab();
            emit("cm_vector(");
            emit(vname);
            emit(", uint, ");
            emit(vector_size);
            emit(", 0, ");
            emit(std::to_string(stride));
            emit(");\n");
            tv.vector_params.insert(vname);
          }
        }
      }
    }

    auto load_exp = std::dynamic_pointer_cast<sem::LoadExpr>(n.init);
    auto s = GetGlobalVarWithOffset(n.init);
    if (load_exp && s.size() > 0) {
      EmitVector(ty, vector_size, n.name);
      emit(";\n");

      EmitReadStat(n.name, n.init);

      CheckValidType(ty);
      scope_->Bind(n.name, ty);
      return;
    }

    if (n.type.array) {
      EmitVector(ty, n.type.array * vector_size, n.name);
      emit(";\n");

      emitTab();
      emit(n.name);
      emit(" = ");
      n.init->Accept(*this);
      emit(";\n");
      CheckValidType(ty);
      scope_->Bind(n.name, ty);
      return;
    }

    auto binary_exp = std::dynamic_pointer_cast<sem::BinaryExpr>(n.init);
    if (binary_exp) {
      if (binary_exp->op == ">" || binary_exp->op == "<" || binary_exp->op == ">=" || binary_exp->op == "<=" ||
          binary_exp->op == "==" || binary_exp->op == "!=") {
        ty.dtype = DataType::INT8;
      }
    }

    auto cond_exp = std::dynamic_pointer_cast<sem::CondExpr>(n.init);
    auto select_exp = std::dynamic_pointer_cast<sem::SelectExpr>(n.init);
    std::string op = (cond_exp || select_exp) ? "." : " = ";

    if (IsVector(n.init)) {
      EmitVector(ty, vector_size, n.name);
      emit(";\n");

      emitTab();
      emit(n.name);
      emit(op);
      n.init->Accept(*this);
      emit(";\n");
    } else {
      emitTab();
      emitType(ty);
      emit(" ");
      emit(n.name);
      emit(op);
      n.init->Accept(*this);
      emit(";\n");
    }

    CheckValidType(ty);
    scope_->Bind(n.name, ty);
    return;
  }

  if (n.type.array) {
    EmitVector(ty, n.type.array * vector_size, n.name);
    emit(" = 0;\n");
  } else {
    emitTab();
    emitType(ty);
    emit(" ");
    emit(n.name);
    emit(";\n");
  }

  CheckValidType(ty);
  scope_->Bind(n.name, ty);
}

void Emit::Visit(const sem::SubscriptLVal& n) {
  if (write_mode) {
    if (!single_element_rw_mode) {
      int stride = GetIndexStride(n.offset);
      if (stride > 1) {
        n.ptr->Accept(*this);
        emit(", ");
        n.offset->Accept(*this);
        emit(", ");
        emit("element_offset_");
        emit(std::to_string(stride));
        return;
      }
    }
    n.ptr->Accept(*this);
    emit(", sizeof(");
    emitType(write_type);
    emit(") * ");
    n.offset->Accept(*this);
    if (IsVector(n.offset)) {
      emit("(0)");
    }
    return;
  }

  if (read_mode) {
    auto s = GetGlobalVarWithOffset(n);
    if (s.length() > 0 && input_replace_map.find(s) != input_replace_map.end()) {
      emit(input_replace_map[s]);
      return;
    }
    n.ptr->Accept(*this);
    auto is_lookup_lval = std::dynamic_pointer_cast<sem::LookupLVal>(n.ptr);
    emit(".select<");
    emit(vector_size);
    emit(", 1>");
    emit("(");
    n.offset->Accept(*this);
    if (large_sparse_vactor.find(is_lookup_lval->name) == large_sparse_vactor.end()) {
      emit(" * ");
      emit(vector_size);
    }
    emit(")");
    return;
  }

  auto s = GetGlobalVarWithOffset(n);
  if (s.length() > 0 && input_replace_map.find(s) != input_replace_map.end()) {
    emit(input_replace_map[s]);
    return;
  }

  n.ptr->Accept(*this);
  auto is_lookup_lval = std::dynamic_pointer_cast<sem::LookupLVal>(n.ptr);

  if (!sub_group_broadcast_first_val) {
    if (vector_stride_map.find(GetLValueName(n.ptr)) == vector_stride_map.end() ||
        vector_stride_map[GetLValueName(n.ptr)] >= 1) {
      emit(".select<");
      emit(vector_size);
      emit(", 1>");
    }
  }

  emit("(");
  n.offset->Accept(*this);
  if (large_sparse_vactor.find(is_lookup_lval->name) == large_sparse_vactor.end()) {
    emit(" * ");
    emit(vector_size);
  }
  if (!sub_group_broadcast_first_val) {
    emit(")");
  }
}

void Emit::Visit(const sem::ClampExpr& n) {
  auto ty_val = TypeOf(n.val);
  auto ty_min = TypeOf(n.min);
  auto ty_max = TypeOf(n.max);

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

  emit("_cmamp(");
  auto load_expr = std::dynamic_pointer_cast<sem::LoadExpr>(n.val);
  if (load_expr) {
    auto s = GetGlobalVarWithOffset(load_expr->inner);
    if (s.length() > 0 && input_replace_map.find(s) != input_replace_map.end()) {
      emit(input_replace_map[s]);

      auto subscript_lval = std::dynamic_pointer_cast<sem::SubscriptLVal>(load_expr->inner);
      if (subscript_lval) {
        emit("(_mod(");
        subscript_lval->offset->Accept(*this);
        emit(" , 4))");
      }

      emit(", ");
      n.min->Accept(*this);
      emit(", ");
      n.max->Accept(*this);
      emit(")");
      return;
    }
  }
  n.val->Accept(*this);
  emit(", ");
  n.min->Accept(*this);
  emit(", ");
  n.max->Accept(*this);
  emit(")");
}

void Emit::Visit(const sem::IndexExpr& n) {
  switch (n.type) {
    case sem::IndexExpr::GLOBAL:
      if (single_eu_mode) {
        emit("_i" + std::to_string(n.dim));
        return;
      }
      emit("(cm_local_size(" + std::to_string(n.dim) + ")");
      emit(" * cm_group_id(" + std::to_string(n.dim) + ")");
      emit(" + cm_local_id(" + std::to_string(n.dim) + "))");
      break;
    case sem::IndexExpr::GROUP:
      emit("cm_group_id(" + std::to_string(n.dim) + ")");
      break;
    case sem::IndexExpr::LOCAL:
      if (single_eu_mode) {
        emit("_i" + std::to_string(n.dim));
        return;
      }
      if (single_element_rw_mode) {
        emit("cm_local_id(" + std::to_string(n.dim) + ")");
      } else {
        emit(vector_size);
        emit(" * cm_local_id(" + std::to_string(n.dim) + ")");
      }
      break;
    default:
      throw std::runtime_error("Invalid IndexExpr type");
  }
}

void Emit::Visit(const sem::Function& n) {
  emit("extern \"C\" _GENX_MAIN_ ");

  if (n.subgroup_size) {
    single_element_rw_mode = false;
    vector_size = n.subgroup_size;
  } else {
    single_element_rw_mode = true;
    vector_size = 4;
  }

  lang::Scope<sem::Type> scope;
  scope_ = &scope;

  single_eu_mode = false;
  int param_index = 0;
  for (const auto& p : n.params) {
    input_params_map[p.second] = param_index;
    param_index++;
    auto ty = p.first;
    if (ty.dtype == DataType::BOOLEAN) {
      ty.dtype = DataType::INT8;
    }
    if (ty.dtype == DataType::INT8 || ty.dtype == DataType::UINT8 || ty.dtype == DataType::INT16 ||
        ty.dtype == DataType::UINT16) {
      single_eu_mode = true;
    }
    CheckValidType(ty);
    scope.Bind(p.second, ty);
    tv.global_params.insert(p.second);
    tv.vector_params.insert(p.second);
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
    emit("SurfaceIndex");
    emit(" ");
    emit(p.second);
  }
  emit(")\n");

  if (single_eu_mode) {
    int g0 = ki_.gwork[0];
    int g1 = ki_.gwork[1];
    int g2 = ki_.gwork[2];
    emit("{\n");
    ++indent_;
    emitTab();
    emit("if(cm_local_id(0) == 0 && cm_group_id(0) == 0){\n");
    ++indent_;
    emitTab();
    emit("for(int _i0=0;_i0<" + std::to_string(g0) + ";_i0++){\n");
    ++indent_;
    emitTab();
    emit("for(int _i1=0;_i1<" + std::to_string(g1) + ";_i1++){\n");
    ++indent_;
    emitTab();
    emit("for(int _i2=0;_i2<" + std::to_string(g2) + ";_i2++){\n");
    n.body->Accept(*this);

    for (int i = 0; i < 5; i++) {
      emitTab();
      emit("}\n");
      --indent_;
    }
  } else {
    n.body->Accept(*this);
  }
  scope_ = nullptr;
}

void Emit::Visit(const sem::CondExpr& n) {
  auto type = TypeOf(n.cond);
  emit("merge(");
  n.tcase->Accept(*this);
  emit(", ");
  n.fcase->Accept(*this);
  emit(", ");

  if (IsVector(n.cond)) {
    emit("vector<ushort,");
    emit(vector_size);
    emit(">(");
    n.cond->Accept(*this);
    emit(")");
  } else {
    n.cond->Accept(*this);
  }

  emit(")");
}

void Emit::Visit(const sem::SelectExpr& n) {
  auto type = TypeOf(n.cond);
  emit("merge(");
  n.tcase->Accept(*this);
  emit(", ");
  n.fcase->Accept(*this);
  emit(", ");

  if (IsVector(n.cond)) {
    emit("vector<ushort,");
    emit(vector_size);
    emit(">(");
    n.cond->Accept(*this);
    emit(")");
  } else {
    n.cond->Accept(*this);
  }

  emit(")");
}

void Emit::Visit(const sem::CastExpr& n) {
  // Since cast is not allowed for cm_vector, basic types casts should be added to anywhere needed.
  n.val->Accept(*this);
}

void Emit::Visit(const sem::CallExpr& n) {
  if (n.name == "sub_group_broadcast") {
    sub_group_broadcast_first_val = true;
    n.vals[0]->Accept(*this);
    sub_group_broadcast_first_val = false;
    emit(" + ");
    n.vals[1]->Accept(*this);
    emit(")");
    return;
  }
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

void Emit::Visit(const sem::BarrierStmt& n) {}

inline std::string c_dtype(const DataType& dt) {
  std::string base;
  switch (dt) {
    case DataType::BOOLEAN:
      base = "bool";
      break;
    case DataType::INT8:
      base = "char";
      break;
    case DataType::INT16:
      base = "short";
      break;
    case DataType::INT32:
      base = "int";
      break;
    case DataType::INT64:
      base = "long";
      break;
    case DataType::UINT8:
      base = "uchar";
      break;
    case DataType::UINT16:
      base = "ushort";
      break;
    case DataType::UINT32:
      base = "uint";
      break;
    case DataType::UINT64:
      base = "ulong";
      break;
    case DataType::FLOAT16:
      base = "half";
      break;
    case DataType::FLOAT32:
      base = "float";
      break;
    case DataType::FLOAT64:
      base = "double";
      break;
    default:
      throw std::runtime_error("Invalid tile type");
  }
  return base;
}

void Emit::EmitReadStat(const sem::LValPtr& lhs, const sem::ExprPtr& rhs) {
  emitTab();
  emit("_read(");
  read_mode = true;
  rhs->Accept(*this);
  emit(", ");
  lhs->Accept(*this);
  emit(");\n");
  read_mode = false;
}

void Emit::EmitReadStat(const std::string& lhs, const sem::ExprPtr& rhs) {
  emitTab();
  emit("_read(");
  read_mode = true;
  rhs->Accept(*this);
  emit(", ");
  emit(lhs);
  emit(");\n");
  read_mode = false;
}

void Emit::EmitWriteStat(const sem::LValPtr& lhs, const sem::ExprPtr& rhs) {
  emitTab();
  emit("_write(");
  write_mode = true;
  write_type = TypeOf(lhs);
  lhs->Accept(*this);
  write_mode = false;
  emit(", ");
  rhs->Accept(*this);
  emit(");\n");

  auto s = GetLValueName(lhs);
  if (input_params_map.find(s) != input_params_map.end()) {
    output_index.insert(input_params_map[s]);
  }
}

void Emit::EmitWriteStat(const sem::LValPtr& lhs, const std::string& rhs) {
  emitTab();
  emit("_write(");
  write_mode = true;
  write_type = TypeOf(lhs);
  lhs->Accept(*this);
  write_mode = false;
  emit(", ");
  emit(rhs);
  emit(");\n");

  auto s = GetLValueName(lhs);
  if (input_params_map.find(s) != input_params_map.end()) {
    output_index.insert(input_params_map[s]);
  }
}

void Emit::EmitSingleElementWriteStat(const sem::LValPtr& lhs, const sem::ExprPtr& rhs) {
  emitTab();
  auto ty_lhs = TypeOf(lhs);
  auto ty_rhs = TypeOf(rhs);
  auto dty_lhs = c_dtype(ty_lhs.dtype);
  auto dty_rhs = c_dtype(ty_rhs.dtype);

  switch (ty_lhs.dtype) {
    case DataType::INT8:
    case DataType::UINT8:
    case DataType::INT16:
    case DataType::UINT16:
    case DataType::FLOAT16:
      emit("_write_single_element(");
      break;
    case DataType::INT32:
    case DataType::UINT32:
    case DataType::FLOAT32:
      emit("_write_atomic_single_dword(");
      break;
    case DataType::INT64:
      emit("_write_atomic_single_long(");
      break;
    default:
      throw std::runtime_error("EmitSingleElementWriteStat: this data type is not supported!");
  }
  write_mode = true;
  write_type = ty_lhs;
  lhs->Accept(*this);
  write_mode = false;
  emit(", ");

  if (dty_lhs != dty_rhs) {
    emit("(" + dty_lhs + ")");
  }

  rhs->Accept(*this);
  if (IsVector(rhs)) {
    emit("(0)");
  }
  emit(");\n");

  auto s = GetLValueName(lhs);
  if (input_params_map.find(s) != input_params_map.end()) {
    output_index.insert(input_params_map[s]);
  }
}

void Emit::emit(int n) { emit(std::to_string(n)); }

void Emit::emit(size_t size) { emit(std::to_string(size)); }

void Emit::AssignGlobalVarToTemp(const sem::ExprPtr& e) {
  auto result_map = GetGlobalLoadExprMap(e);
  for (auto result : result_map) {
    std::string temp = "cm_temp" + std::to_string(temp_num);
    temp_num++;
    auto ty = TypeOf(result.first->inner);
    EmitVector(ty, vector_size, temp);
    emit(";\n");

    auto s = GetGlobalVarWithOffset(result.first->inner);
    if (s.length() > 0 && input_replace_map.find(s) != input_replace_map.end()) {
      emitTab();
      emit(temp);
      emit(" = ");
      emit(input_replace_map[s]);
      emit(";\n");
    } else {
      EmitReadStat(temp, result.first);
      input_replace_map[result.second] = temp;
    }
  }
}

std::string Emit::GetLValueName(const sem::LValPtr& p) {
  auto lookup_lval = std::dynamic_pointer_cast<sem::LookupLVal>(p);
  if (lookup_lval) {
    return lookup_lval->name;
  }

  auto subscript_lval = std::dynamic_pointer_cast<sem::SubscriptLVal>(p);
  if (subscript_lval) {
    return GetLValueName(subscript_lval->ptr);
  }

  throw error::Unimplemented{"GetLValueName: Not Supported LValue"};
}

void Emit::CheckValidType(const sem::Type& ty) {
  if (ty.base == sem::Type::TVOID || ty.base == sem::Type::INDEX) {
    return;
  }
  if (ty.dtype == DataType::FLOAT64) {
    throw error::Unimplemented{"The device does not support 64-bit floating-point types"};
  }
}

sem::Type Emit::TypeOf(const sem::ExprPtr& p) { return lang::ExprType::TypeOf(scope_, false, true, p); }

sem::Type Emit::TypeOf(const sem::LValPtr& p) { return lang::ExprType::TypeOf(scope_, false, true, p); }

bool Emit::IsVector(const sem::ExprPtr& p) {
  tv.InitCheckVector();
  p->Accept(tv);
  return tv.CheckVector();
}

bool Emit::IsVector(const sem::LValPtr& p) {
  tv.InitCheckVector();
  p->Accept(tv);
  return tv.CheckVector();
}

bool Emit::IsVector(const sem::LValue& v) {
  tv.InitCheckVector();
  v.Accept(tv);
  return tv.CheckVector();
}

int Emit::GetIndexStride(const sem::ExprPtr& p) {
  tv.InitIndexStride();
  p->Accept(tv);
  return tv.GetIndexStride();
}

int Emit::GetIndexStride(const sem::LValPtr& p) {
  tv.InitIndexStride();
  p->Accept(tv);
  return tv.GetIndexStride();
}

int Emit::GetIndexStride(const sem::LValue& v) {
  tv.InitIndexStride();
  v.Accept(tv);
  return tv.GetIndexStride();
}

std::string Emit::GetGlobalVarWithOffset(const sem::ExprPtr& p) {
  tv.InitGlobalVarWithOffset();
  p->Accept(tv);
  return tv.GetGlobalVarWithOffset();
}

std::string Emit::GetGlobalVarWithOffset(const sem::LValPtr& p) {
  tv.InitGlobalVarWithOffset();
  p->Accept(tv);
  return tv.GetGlobalVarWithOffset();
}

std::string Emit::GetGlobalVarWithOffset(const sem::LValue& v) {
  tv.InitGlobalVarWithOffset();
  v.Accept(tv);
  return tv.GetGlobalVarWithOffset();
}

std::map<std::shared_ptr<sem::LoadExpr>, std::string> Emit::GetGlobalLoadExprMap(const sem::ExprPtr& p) {
  tv.InitGlobalLoadExprMap();
  p->Accept(tv);
  return tv.GetGlobalLoadExprMap();
}

void Emit::EmitVector(const sem::Type& type, const size_t& size, const std::string& name) {
  if (size >= 2048) {
    large_sparse_vactor.insert(name);
    EmitVector(type, size / vector_size, name);
    return;
  }

  emitTab();
  emit("vector<");
  emitType(type);
  emit(",");
  emit(size);
  emit("> ");
  emit(name);
  tv.vector_params.insert(name);
}

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
