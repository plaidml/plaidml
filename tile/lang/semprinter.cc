
#include "tile/lang/semprinter.h"

#include <map>
#include <utility>

#include "tile/lang/fpconv.h"

namespace vertexai {
namespace tile {
namespace sem {

using lang::DataType;

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

void Print::emitType(const Type& t) {
  if (t.base == Type::TVOID) {
    emit("void");
    return;
  }
  if (t.base == Type::INDEX) {
    emit("int");
    return;
  }
  if (t.base == Type::POINTER_CONST) emit("const ");
  emit(c_dtype(t.dtype));
  if (t.vec_width > 1) emit(std::to_string(t.vec_width));
  if (t.base == Type::POINTER_MUT || t.base == Type::POINTER_CONST) {
    emit("*");
  }
}

void Print::Visit(const IntConst& n) { emit(std::to_string(n.value)); }

void Print::Visit(const FloatConst& n) {
  std::string c = lang::DoubleToString(n.value);
  if (c.find_first_of(".e") == std::string::npos) {
    c += ".0";
  }
  emit(c + "f");
}

void Print::Visit(const LookupLVal& n) { emit(n.name); }

void Print::Visit(const LoadExpr& n) { n.inner->Accept(*this); }

void Print::Visit(const StoreStmt& n) {
  emitTab();
  n.lhs->Accept(*this);
  emit(" = ");
  n.rhs->Accept(*this);
  emit(";\n");
}

void Print::Visit(const SubscriptLVal& n) {
  n.ptr->Accept(*this);
  emit("[");
  n.offset->Accept(*this);
  emit("]");
}

void Print::Visit(const DeclareStmt& n) {
  emitTab();
  emitType(n.type);
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
        if (i > 0) {
          emit(", ");
        }
        n.init->Accept(*this);
      }
      emit("}");
    } else {
      n.init->Accept(*this);
    }
  }
  emit(";\n");
}

void Print::Visit(const UnaryExpr& n) {
  emit("(");
  emit(n.op);
  n.inner->Accept(*this);
  emit(")");
}

void Print::Visit(const BinaryExpr& n) {
  emit("(");
  n.lhs->Accept(*this);
  emit(" " + n.op + " ");
  n.rhs->Accept(*this);
  emit(")");
}

void Print::Visit(const CondExpr& n) {
  emit("(");
  n.cond->Accept(*this);
  emit("? ");
  n.tcase->Accept(*this);
  emit(": ");
  n.fcase->Accept(*this);
  emit(")");
}

void Print::Visit(const SelectExpr& n) {
  emit("(");
  n.cond->Accept(*this);
  emit("?? ");
  n.tcase->Accept(*this);
  emit(": ");
  n.fcase->Accept(*this);
  emit(")");
}

void Print::Visit(const ClampExpr& n) {
  emit("clamp(");
  n.val->Accept(*this);
  emit(", ");
  n.min->Accept(*this);
  emit(", ");
  n.max->Accept(*this);
  emit(")");
}

void Print::Visit(const CastExpr& n) {
  emit("((");
  emitType(n.type);
  emit(") ");
  n.val->Accept(*this);
  emit(")");
}

void Print::Visit(const CallExpr& n) {
  n.func->Accept(*this);
  emit("(");
  for (size_t i = 0; i < n.vals.size(); i++) {
    n.vals[i]->Accept(*this);
    if (i != n.vals.size() - 1) {
      emit(", ");
    }
  }
  emit(")");
}

static std::map<std::pair<DataType, LimitConst::Which>, std::string> LimitConstLookup = {
    {{DataType::BOOLEAN, LimitConst::MIN}, "0"},        {{DataType::INT8, LimitConst::MIN}, "SCHAR_MIN"},
    {{DataType::INT16, LimitConst::MIN}, "SHRT_MIN"},   {{DataType::INT32, LimitConst::MIN}, "INT_MIN"},
    {{DataType::INT64, LimitConst::MIN}, "LONG_MIN"},   {{DataType::UINT8, LimitConst::MIN}, "0"},
    {{DataType::UINT16, LimitConst::MIN}, "0"},         {{DataType::UINT32, LimitConst::MIN}, "0"},
    {{DataType::UINT64, LimitConst::MIN}, "0"},         {{DataType::FLOAT16, LimitConst::MIN}, "-0x1.ffcp15h"},
    {{DataType::FLOAT32, LimitConst::MIN}, "-FLT_MAX"}, {{DataType::FLOAT64, LimitConst::MIN}, "-DBL_MAX"},

    {{DataType::BOOLEAN, LimitConst::MAX}, "0"},        {{DataType::INT8, LimitConst::MAX}, "SCHAR_MAX"},
    {{DataType::INT16, LimitConst::MAX}, "SHRT_MAX"},   {{DataType::INT32, LimitConst::MAX}, "INT_MAX"},
    {{DataType::INT64, LimitConst::MAX}, "LONG_MAX"},   {{DataType::UINT8, LimitConst::MAX}, "UCHAR_MAX"},
    {{DataType::UINT16, LimitConst::MAX}, "USHRT_MAX"}, {{DataType::UINT32, LimitConst::MAX}, "UINT_MAX"},
    {{DataType::UINT64, LimitConst::MAX}, "ULONG_MAX"}, {{DataType::FLOAT16, LimitConst::MAX}, "0x1.ffcp15h"},
    {{DataType::FLOAT32, LimitConst::MAX}, "FLT_MAX"},  {{DataType::FLOAT64, LimitConst::MAX}, "DBL_MAX"},
};

void Print::Visit(const LimitConst& n) {
  if (n.which == LimitConst::ZERO) {
    emit("0");
    return;
  }
  auto it = LimitConstLookup.find(std::make_pair(n.type, n.which));
  if (it == LimitConstLookup.end()) {
    throw std::runtime_error("Invalid type in LimitConst");
  }
  emit(it->second);
}

void Print::Visit(const IndexExpr& n) {
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

void Print::Visit(const Block& n) {
  emitTab();
  emit("{\n");
  ++indent_;
  for (const StmtPtr& ptr : n.statements) {
    ptr->Accept(*this);
  }
  --indent_;
  emitTab();
  emit("}\n");
}

void Print::Visit(const IfStmt& n) {
  emitTab();
  if (n.iftrue && n.iffalse) {
    emit("if (");
    n.cond->Accept(*this);
    emit(")\n");
    n.iftrue->Accept(*this);
    emitTab();
    emit("else\n");
    n.iffalse->Accept(*this);
  } else if (n.iftrue) {
    emit("if (");
    n.cond->Accept(*this);
    emit(")\n");
    n.iftrue->Accept(*this);
  } else if (n.iffalse) {
    emit("if (!");
    n.cond->Accept(*this);
    emit(")\n");
    n.iffalse->Accept(*this);
  }
}

void Print::Visit(const ForStmt& n) {
  emitTab();
  emit("for(int ");
  emit(n.var);
  emit(" = 0; ");
  emit(n.var);
  emit(" < ");
  emit(std::to_string(n.num * n.step));
  emit("; ");
  emit(n.var);
  emit(" += ");
  emit(std::to_string(n.step));
  emit(")\n");
  n.inner->Accept(*this);
}

void Print::Visit(const WhileStmt& n) {
  emitTab();
  emit("while (");
  n.cond->Accept(*this);
  emit(")\n");
  n.inner->Accept(*this);
}

void Print::Visit(const BarrierStmt& n) {
  emitTab();
  emit("barrier();\n");
}

void Print::Visit(const ReturnStmt& n) {
  emitTab();
  emit("return");
  if (n.value) {
    emit(" (");
    n.value->Accept(*this);
    emit(")");
  }
  emit(";\n");
}

void Print::Visit(const Function& n) {
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
    emitType(p.first);
    emit(" ");
    emit(p.second);
  }
  emit(")\n");
  n.body->Accept(*this);
}

}  // namespace sem
}  // namespace tile
}  // namespace vertexai
