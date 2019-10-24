// Copyright 2018, Intel Corporation.

#include <utility>

#include "tile/lang/exprtype.h"
#include "tile/lang/fpconv.h"
#include "tile/lang/generate.h"
#include "tile/lang/scope.h"

#define QUAD_GROUP

#if defined(SIMD_GROUP) && defined(QUAD_GROUP)
#error Cannot define both SIMD_GROUP and QUAD_GROUP.
#endif

#ifdef SIMD_GROUP
#define BROADCAST "simd_broadcast"
#define BARRIER "simdgroup_barrier"
#endif

#ifdef QUAD_GROUP
#define BROADCAST "quad_broadcast"
#define BARRIER "simdgroup_barrier"
#endif

#if !defined(BROADCAST)
#error BROADCAST is not defined.
#endif

namespace vertexai {
namespace tile {
namespace hal {
namespace metal {

inline std::string c_dtype(const DataType& dt) {
  switch (dt) {
    case DataType::BOOLEAN:
      return "bool";
    case DataType::INT8:
      return "char";
    case DataType::INT16:
      return "short";
    case DataType::INT32:
      return "int";
    case DataType::INT64:
      return "ptrdiff_t";
    case DataType::UINT8:
      return "uchar";
    case DataType::UINT16:
      return "ushort";
    case DataType::UINT32:
      return "uint";
    case DataType::UINT64:
      return "size_t";
    case DataType::FLOAT16:
      return "half";
    case DataType::FLOAT32:
      return "float";
    case DataType::FLOAT64:
    default:
      throw std::runtime_error{"Unusable hardware type: " + to_string(dt)};
  }
}

static std::map<std::pair<DataType, sem::LimitConst::Which>, std::string> LimitConstLookup = {
    {{DataType::BOOLEAN, sem::LimitConst::MIN}, "0"},        {{DataType::INT8, sem::LimitConst::MIN}, "SCHAR_MIN"},
    {{DataType::INT16, sem::LimitConst::MIN}, "SHRT_MIN"},   {{DataType::INT32, sem::LimitConst::MIN}, "INT_MIN"},
    {{DataType::INT64, sem::LimitConst::MIN}, "LONG_MIN"},   {{DataType::UINT8, sem::LimitConst::MIN}, "0"},
    {{DataType::UINT16, sem::LimitConst::MIN}, "0"},         {{DataType::UINT32, sem::LimitConst::MIN}, "0"},
    {{DataType::UINT64, sem::LimitConst::MIN}, "0"},         {{DataType::FLOAT16, sem::LimitConst::MIN}, "-65504"},
    {{DataType::FLOAT32, sem::LimitConst::MIN}, "-FLT_MAX"}, {{DataType::FLOAT64, sem::LimitConst::MIN}, "-DBL_MAX"},

    {{DataType::BOOLEAN, sem::LimitConst::MAX}, "0"},        {{DataType::INT8, sem::LimitConst::MAX}, "SCHAR_MAX"},
    {{DataType::INT16, sem::LimitConst::MAX}, "SHRT_MAX"},   {{DataType::INT32, sem::LimitConst::MAX}, "INT_MAX"},
    {{DataType::INT64, sem::LimitConst::MAX}, "LONG_MAX"},   {{DataType::UINT8, sem::LimitConst::MAX}, "UCHAR_MAX"},
    {{DataType::UINT16, sem::LimitConst::MAX}, "USHRT_MAX"}, {{DataType::UINT32, sem::LimitConst::MAX}, "UINT_MAX"},
    {{DataType::UINT64, sem::LimitConst::MAX}, "ULONG_MAX"}, {{DataType::FLOAT16, sem::LimitConst::MAX}, "65504"},
    {{DataType::FLOAT32, sem::LimitConst::MAX}, "FLT_MAX"},  {{DataType::FLOAT64, sem::LimitConst::MAX}, "DBL_MAX"},
};

class Emitter : public sem::Visitor {
 public:
  std::string str() const {  //
    return result_.str();
  }

  void Visit(const sem::LimitConst& node) {
    if (node.which == sem::LimitConst::ZERO) {
      emit("0");
      return;
    } else if (node.which == sem::LimitConst::ONE) {
      emit("1");
      return;
    }
    auto it = LimitConstLookup.find(std::make_pair(node.type, node.which));
    if (it == LimitConstLookup.end()) {
      throw std::runtime_error("Invalid type in LimitConst");
    }
    emit(it->second);
  }

  void Visit(const sem::IntConst& node) {  //
    emit(std::to_string(node.value));
  }

  void Visit(const sem::FloatConst& node) {
    std::string c = lang::DoubleToString(node.value);
    if (c.find_first_of(".e") == std::string::npos) {
      c += ".0";
    }
    emit(c + "f");
  }

  void Visit(const sem::LookupLVal& node) {  //
    emit(node.name);
  }

  void Visit(const sem::LoadExpr& node) {  //
    node.inner->Accept(*this);
  }

  void Visit(const sem::StoreStmt& node) {
    emitTab();
    node.lhs->Accept(*this);
    emit(" = ");
    node.rhs->Accept(*this);
    emit(";\n");
  }

  void Visit(const sem::SubscriptLVal& node) {
    node.ptr->Accept(*this);
    emit("[");
    node.offset->Accept(*this);
    emit("]");
  }

  void Visit(const sem::UnaryExpr& node) {
    emit("(");
    emit(node.op);
    node.inner->Accept(*this);
    emit(")");
  }

  void Visit(const sem::BinaryExpr& node) {
    auto lhs_type = TypeOf(node.lhs);
    auto rhs_type = TypeOf(node.rhs);
    auto tgt_type = lang::Promote({lhs_type, rhs_type});
    emit("(");
    EmitWithTypeConversion(lhs_type, tgt_type, node.lhs, false);
    emit(" ");
    emit(node.op);
    emit(" ");
    EmitWithTypeConversion(rhs_type, tgt_type, node.rhs, false);
    emit(")");
  }

  void Visit(const sem::CondExpr& node) {  //
    Select(node.type, node.cond, node.tcase, node.fcase);
  }

  void Visit(const sem::SelectExpr& node) {  //
    Select(node.type, node.cond, node.tcase, node.fcase);
  }

  void Visit(const sem::Block& node) {
    auto previous_scope = scope_;
    lang::Scope<sem::Type> scope{scope_};
    scope_ = &scope;
    emitTab();
    emit("{\n");
    ++indent_;
    if (initial_block_) {
      initial_block_ = false;
      for (size_t i = 0; i < params_.size(); i++) {
        const auto& item = params_[i];
        emitTab();
        emitType(item.first);
        emit(" ");
        emit(item.second);
        emit(" = ");
        emit("static_cast<");
        emitType(item.first);
        emit(">(");
        emit(item.second);
        emit("_arg_);\n");
      }
    }
    for (const sem::StmtPtr& ptr : node.statements) {
      ptr->Accept(*this);
    }
    --indent_;
    emitTab();
    emit("}\n");
    scope_ = previous_scope;
  }

  void Visit(const sem::ClampExpr& node) {
    emit("clamp(");
    node.val->Accept(*this);
    emit(", ");
    node.min->Accept(*this);
    emit(", ");
    node.max->Accept(*this);
    emit(")");
  }

  void Visit(const sem::CallExpr& node) {
    switch (node.function) {
      case sem::CallExpr::Function::CEIL:
      case sem::CallExpr::Function::FLOOR: {
        assert(1 == node.vals.size());
        emit(node.name);
        emit("(");
        auto val_type = TypeOf(node.vals[0]);
        auto need_type = val_type;
        switch (need_type.dtype) {
          case DataType::BOOLEAN:
          case DataType::INT8:
          case DataType::INT16:
          case DataType::UINT8:
          case DataType::UINT16:
            need_type.dtype = DataType::FLOAT16;
            break;
          case DataType::INT32:
          case DataType::UINT32:
            need_type.dtype = DataType::FLOAT32;
            break;
          case DataType::INT64:
          case DataType::UINT64:
            need_type.dtype = DataType::FLOAT64;
            break;
          default:
            break;
        }
        EmitWithTypeConversion(val_type, need_type, node.vals[0], false);
        emit(")");
      } break;
      default: {
        if (node.name == "sub_group_broadcast") {
          emit(BROADCAST);
        } else {
          emit(node.name);
        }
        emit("(");
        for (size_t i = 0; i < node.vals.size(); i++) {
          if (i) {
            emit(", ");
          }
          node.vals[i]->Accept(*this);
        }
        emit(")");
      } break;
    }
  }

  void Visit(const sem::DeclareStmt& node) {
    emitTab();
    emitType(node.type);
    emit(" ");
    emit(node.name);
    if (node.type.array) {
      emit("[" + std::to_string(node.type.array) + "]");
    }
    if (node.init) {
      emit(" = ");
      if (node.type.array) {
        emit("{");
        for (size_t i = 0; i < node.type.array; i++) {
          node.init->Accept(*this);
          emit(", ");
        }
        emit("}");
      } else {
        EmitWithTypeConversion(TypeOf(node.init), node.type, node.init, false);
      }
    }
    emit(";\n");
    scope_->Bind(node.name, node.type);
  }

  void Visit(const sem::IfStmt& node) {
    emitTab();
    if (node.iftrue && node.iffalse) {
      emit("if (");
      node.cond->Accept(*this);
      emit(")\n");
      node.iftrue->Accept(*this);
      emitTab();
      emit("else\n");
      node.iffalse->Accept(*this);
    } else if (node.iftrue) {
      emit("if (");
      node.cond->Accept(*this);
      emit(")\n");
      node.iftrue->Accept(*this);
    } else if (node.iffalse) {
      // This code is required since it is possible for node.iftrue to be a nullptr.
      // It needs to stay in place because its possible for verbose logging to print
      // pre-simplified code; this would cause a null pointer to be dereferencd and hence a crash.
      emit("if !(");
      node.cond->Accept(*this);
      emit(")\n");
      node.iffalse->Accept(*this);
    }
  }

  void Visit(const sem::ForStmt& node) {
    auto previous_scope = scope_;
    lang::Scope<sem::Type> scope{scope_};
    scope_ = &scope;
    scope.Bind(node.var, sem::Type{sem::Type::INDEX});
    emitTab();
    emit("for (int ");
    emit(node.var);
    emit(" = 0; ");
    emit(node.var);
    emit(" < ");
    emit(std::to_string(node.num * node.step));
    emit("; ");
    emit(node.var);
    emit(" += ");
    emit(std::to_string(node.step));
    emit(")\n");
    node.inner->Accept(*this);
    scope_ = previous_scope;
  }

  void Visit(const sem::WhileStmt& node) {
    emitTab();
    emit("while (");
    node.cond->Accept(*this);
    emit(")\n");
    node.inner->Accept(*this);
  }

  void Visit(const sem::CastExpr& node) { node.val->Accept(*this); }

  void Visit(const sem::IndexExpr& node) {
    switch (node.type) {
      case sem::IndexExpr::GLOBAL:
        emit("_globalid[" + std::to_string(node.dim) + "]");
        break;
      case sem::IndexExpr::GROUP:
        emit("_groupid[" + std::to_string(node.dim) + "]");
        break;
      case sem::IndexExpr::LOCAL:
        emit("_tid");
        break;
      default:
        throw std::runtime_error("Invalid IndexExpr type");
    }
  }

  void Visit(const sem::BarrierStmt& node) {
    emitTab();
    if (node.subgroup) {
      emit(std::string(BARRIER) + std::string("(mem_flags::mem_threadgroup);\n"));
    } else {
      emit("threadgroup_barrier(mem_flags::mem_threadgroup);\n");
    }
  }

  void Visit(const sem::ReturnStmt& node) {
    emitTab();
    emit("return");
    if (node.value) {
      emit(" (");
      node.value->Accept(*this);
      emit(")");
    }
    emit(";\n");
  }

  void Visit(const sem::SpecialStmt& node) {
    throw std::runtime_error("Metal code emitter special statement not defined!");
  }

  void Visit(const sem::Function& node) {
    lang::Scope<sem::Type> scope;
    scope_ = &scope;
    emit("kernel ");
    emitType(node.ret);
    emit(" ");
    emit(node.name);
    emit("(\n");
    for (size_t i = 0; i < node.params.size(); i++) {
      const auto& item = node.params[i];
      emit("    ");
      emitType(item.first, true);
      emit(" ");
      emit(item.second);
      emit("_arg_");
      emit(" [[ buffer(" + std::to_string(i) + ") ]],\n");
      scope.Bind(item.second, item.first);
    }
    emit("    uint _tid [[ thread_index_in_threadgroup ]],\n");
    emit("    uint3 _groupid [[ threadgroup_position_in_grid ]],\n");
    emit("    uint3 _globalid [[ thread_position_in_grid ]]\n");
    emit(")\n");
    params_ = node.params;
    node.body->Accept(*this);
    scope_ = nullptr;
  }

 private:
  void Select(const sem::Type& result_type, const sem::ExprPtr& cond, const sem::ExprPtr& tcase, const sem::ExprPtr& fcase) {
    auto tcase_type = TypeOf(tcase);
    auto fcase_type = TypeOf(fcase);
    auto cond_type = TypeOf(cond);
    auto tgt_type = lang::Promote({tcase_type, fcase_type});
    tgt_type.vec_width = std::max(tgt_type.vec_width, cond_type.vec_width);
    const_cast<sem::Type&>(result_type).vec_width = tgt_type.vec_width;
    emit("select(");
    EmitWithTypeConversion(fcase_type, result_type, fcase, true);
    emit(", ");
    EmitWithTypeConversion(tcase_type, result_type, tcase, true);
    emit(", ");
    EmitWithWidthConversion(cond_type, tgt_type, cond, true);
    emit(")");
  }

  void EmitWithTypeConversion(const sem::Type& from,     //
                              const sem::Type& to,       //
                              const sem::ExprPtr& expr,  //
                              bool force_conversion) {
    if (to.base == sem::Type::POINTER_MUT || to.base == sem::Type::POINTER_CONST) {
      // No conversion required for pointer types.
      expr->Accept(*this);
      return;
    }
    if (!force_conversion && (from.vec_width == to.vec_width) &&
        ((from.vec_width == 1 && from.base == sem::Type::VALUE && is_int(from.dtype) &&
          (to.base == sem::Type::INDEX || (to.base == sem::Type::VALUE && is_int(to.dtype)))) ||
         (from.base == to.base && from.dtype == to.dtype))) {
      // No conversion required.
      expr->Accept(*this);
      return;
    }
    emit("(");
    emitType(to);
    emit(")");
    expr->Accept(*this);
  }

  void EmitWithWidthConversion(const sem::Type& from, const sem::Type& to, const sem::ExprPtr& expr,
                               bool force_conversion) {
    if (to.base == sem::Type::POINTER_MUT || to.base == sem::Type::POINTER_CONST) {
      // No conversion required.
      expr->Accept(*this);
      return;
    }

    sem::Type condition_type = to;
    condition_type.dtype = DataType::BOOLEAN;

    EmitWithTypeConversion(from, condition_type, expr, force_conversion);
    if (from.vec_width != to.vec_width) {
      // We need to convert a scalar into a vector.
      emit(" != ");
      emit("(");
      emitType(condition_type);
      emit(")");
      emit("0");
    }
  }

  sem::Type TypeOf(const sem::ExprPtr& expr) {  //
    return lang::ExprType::TypeOf(scope_, true, false, expr);
  }

  sem::Type TypeOf(const sem::LValPtr& lvalue) {  //
    return lang::ExprType::TypeOf(scope_, true, false, lvalue);
  }

  void emitType(const sem::Type& type, bool is_param = false) {
    if (type.region == sem::Type::LOCAL) {
      emit("threadgroup ");
    } else if (type.region == sem::Type::GLOBAL) {
      emit("device ");
    }
    if (type.base == sem::Type::TVOID) {
      emit("void");
      return;
    }
    if (type.base == sem::Type::INDEX) {
      emit("int");
      return;
    }
    if (type.base == sem::Type::POINTER_CONST) {
      emit("const ");
    }
    if (is_param) {
      emit("void");
    } else {
      emit(c_dtype(type.dtype));
      if (type.vec_width > 1) {
        emit(std::to_string(type.vec_width));
      }
    }
    if (type.base == sem::Type::POINTER_MUT || type.base == sem::Type::POINTER_CONST) {
      emit("*");
    }
  }

  void emit(const std::string& s) {  //
    result_ << s;
  }

  void emitTab() {  //
    result_ << std::string(indent_ << 1, ' ');
  }

 private:
  std::ostringstream result_;
  size_t indent_ = 0;
  lang::Scope<sem::Type>* scope_ = nullptr;
  bool initial_block_ = true;
  sem::Function::params_t params_;
};

std::string EmitMetal(const lang::KernelInfo& ki) {
  Emitter emitter;
  emitter.Visit(*ki.kfunc);
  return emitter.str();
}

}  // namespace metal
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
