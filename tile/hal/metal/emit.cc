// Copyright 2018, Vertex.AI.

#include <utility>

#include "tile/hal/metal/hal.h"
#include "tile/lang/emitc.h"
#include "tile/lang/exprtype.h"
#include "tile/lang/scope.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace metal {

inline std::string c_dtype(const lang::DataType& dt) {
  switch (dt) {
    case lang::DataType::BOOLEAN:
      return "bool";
    case lang::DataType::INT8:
      return "char";
    case lang::DataType::INT16:
      return "short";
    case lang::DataType::INT32:
      return "int";
    case lang::DataType::INT64:
      return "ptrdiff_t";
    case lang::DataType::UINT8:
      return "uchar";
    case lang::DataType::UINT16:
      return "ushort";
    case lang::DataType::UINT32:
      return "uint";
    case lang::DataType::UINT64:
      return "size_t";
    case lang::DataType::FLOAT16:
      return "half";
    case lang::DataType::FLOAT32:
      return "float";
    case lang::DataType::FLOAT64:
    default:
      throw std::runtime_error("Invalid tile type");
  }
}

class Emitter : public lang::EmitC {
 public:
  void Visit(const sem::CondExpr& node) final {  //
    Select(node.cond, node.tcase, node.fcase);
  }

  void Visit(const sem::SelectExpr& node) final {  //
    Select(node.cond, node.tcase, node.fcase);
  }

  void Visit(const sem::Block& node) final {
    auto previous_scope = scope_;
    lang::Scope<sem::Type> scope{scope_};
    scope_ = &scope;
    EmitC::Visit(node);
    scope_ = previous_scope;
  }

  void Visit(const sem::DeclareStmt& node) final {
    EmitC::Visit(node);
    scope_->Bind(node.name, node.type);
  }

  void Visit(const sem::ForStmt& node) final {
    auto previous_scope = scope_;
    lang::Scope<sem::Type> scope{scope_};
    scope_ = &scope;
    scope.Bind(node.var, sem::Type{sem::Type::INDEX});
    EmitC::Visit(node);
    scope_ = previous_scope;
  }

  void Visit(const sem::IndexExpr& node) final {
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

  void Visit(const sem::BarrierStmt& node) final {
    emitTab();
    emit("threadgroup_barrier(mem_flags::mem_threadgroup);\n");
  }

  void Visit(const sem::Function& node) final {
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
      emitType(item.first);
      emit(" ");
      emit(item.second);
      emit(" [[ buffer(" + std::to_string(i) + ") ]],\n");
      scope.Bind(item.second, item.first);
    }
    emit("    uint _tid [[ thread_index_in_threadgroup ]],\n");
    emit("    uint3 _groupid [[ threadgroup_position_in_grid ]],\n");
    emit("    uint3 _globalid [[ thread_position_in_grid ]]\n");
    emit(")\n");
    node.body->Accept(*this);

    scope_ = nullptr;
  }

 private:
  void Select(const sem::ExprPtr& cond, const sem::ExprPtr& tcase, const sem::ExprPtr& fcase) {
    auto tcase_type = TypeOf(tcase);
    auto fcase_type = TypeOf(fcase);
    auto cond_type = TypeOf(cond);
    auto tgt_type = lang::Promote({tcase_type, fcase_type});
    tgt_type.vec_width = std::max(tgt_type.vec_width, cond_type.vec_width);
    emit("select(");
    EmitWithTypeConversion(fcase_type, tgt_type, fcase, true);
    emit(", ");
    EmitWithTypeConversion(tcase_type, tgt_type, tcase, true);
    emit(", ");
    cond->Accept(*this);
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
    if (!force_conversion && ((from.vec_width == 1 && from.base == sem::Type::VALUE && is_int(from.dtype) &&
                               (to.base == sem::Type::INDEX || (to.base == sem::Type::VALUE && is_int(to.dtype)))) ||
                              (from.base == to.base && from.dtype == to.dtype && from.vec_width == to.vec_width))) {
      // No conversion required.
      expr->Accept(*this);
      return;
    }
    emit("(");
    EmitC::emitType(to);
    emit(")");
    expr->Accept(*this);
  }

  sem::Type TypeOf(const sem::ExprPtr& expr) {  //
    return lang::ExprType::TypeOf(scope_, true, expr);
  }

  sem::Type TypeOf(const sem::LValPtr& lvalue) {  //
    return lang::ExprType::TypeOf(scope_, true, lvalue);
  }

  void emitType(const sem::Type& type) final {
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
    emit(c_dtype(type.dtype));
    if (type.vec_width > 1) {
      emit(std::to_string(type.vec_width));
    }
    if (type.base == sem::Type::POINTER_MUT || type.base == sem::Type::POINTER_CONST) {
      emit("*");
    }
  }

 private:
  lang::Scope<sem::Type>* scope_ = nullptr;
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
