
#include "tile/lang/emitc.h"

#include "tile/lang/fpconv.h"

namespace vertexai {
namespace tile {
namespace lang {

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

void EmitC::emitType(const sem::Type& t) {
  if (t.base == sem::Type::TVOID) {
    emit("void");
    return;
  }
  if (t.base == sem::Type::INDEX) {
    emit("int");
    return;
  }
  if (t.base == sem::Type::POINTER_CONST) emit("const ");
  emit(c_dtype(t.dtype));
  if (t.vec_width > 1) emit(std::to_string(t.vec_width));
  if (t.base == sem::Type::POINTER_MUT || t.base == sem::Type::POINTER_CONST) {
    emit("*");
  }
}

void EmitC::Visit(const sem::IntConst& n) { emit(std::to_string(n.value)); }

void EmitC::Visit(const sem::FloatConst& n) {
  std::string c = DoubleToString(n.value);
  if (c.find_first_of(".e") == std::string::npos) {
    c += ".0";
  }
  emit(c + "f");
}

void EmitC::Visit(const sem::LookupLVal& n) { emit(n.name); }

void EmitC::Visit(const sem::LoadExpr& n) { n.inner->Accept(*this); }

void EmitC::Visit(const sem::StoreStmt& n) {
  emitTab();
  n.lhs->Accept(*this);
  emit(" = ");
  n.rhs->Accept(*this);
  emit(";\n");
}

void EmitC::Visit(const sem::SubscriptLVal& n) {
  n.ptr->Accept(*this);
  emit("[");
  n.offset->Accept(*this);
  emit("]");
}

void EmitC::Visit(const sem::DeclareStmt& n) {
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
        n.init->Accept(*this);
        emit(", ");
      }
      emit("}");
    } else {
      n.init->Accept(*this);
    }
  }
  emit(";\n");
}

void EmitC::Visit(const sem::UnaryExpr& n) {
  emit("(");
  emit(n.op);
  n.inner->Accept(*this);
  emit(")");
}

void EmitC::Visit(const sem::BinaryExpr& n) {
  emit("(");
  n.lhs->Accept(*this);
  emit(" ");
  emit(n.op);
  emit(" ");
  n.rhs->Accept(*this);
  emit(")");
}

void EmitC::Visit(const sem::CondExpr& n) {
  emit("(");
  n.cond->Accept(*this);
  emit(" ? ");
  n.tcase->Accept(*this);
  emit(" : ");
  n.fcase->Accept(*this);
  emit(")");
}

void EmitC::Visit(const sem::SelectExpr& n) {
  emit("select(");
  n.fcase->Accept(*this);
  emit(", ");
  n.tcase->Accept(*this);
  emit(", ");
  n.cond->Accept(*this);
  emit(")");
}

void EmitC::Visit(const sem::ClampExpr& n) {
  emit("clamp(");
  n.val->Accept(*this);
  emit(", ");
  n.min->Accept(*this);
  emit(", ");
  n.max->Accept(*this);
  emit(")");
}

void EmitC::Visit(const sem::CastExpr& n) {
  emit("((");
  emitType(n.type);
  emit(")");
  n.val->Accept(*this);
  emit(")");
}

void EmitC::Visit(const sem::CallExpr& n) {
  n.func->Accept(*this);
  emit("(");
  for (size_t i = 0; i < n.vals.size(); i++) {
    if (i) {
      emit(", ");
    }
    n.vals[i]->Accept(*this);
  }
  emit(")");
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

void EmitC::Visit(const sem::LimitConst& n) {
  if (n.which == sem::LimitConst::ZERO) {
    emit("0");
    return;
  } else if (n.which == sem::LimitConst::ONE) {
    emit("1");
    return;
  }
  auto it = LimitConstLookup.find(std::make_pair(n.type, n.which));
  if (it == LimitConstLookup.end()) {
    throw std::runtime_error("Invalid type in LimitConst");
  }
  emit(it->second);
}

void EmitC::Visit(const sem::IndexExpr& n) { throw std::runtime_error("IndexExpr unimplemented in EmitC"); }

void EmitC::Visit(const sem::Block& n) {
  emitTab();
  emit("{\n");
  ++indent_;
  for (const sem::StmtPtr& ptr : n.statements) {
    ptr->Accept(*this);
  }
  --indent_;
  emitTab();
  emit("}\n");
}

void EmitC::Visit(const sem::IfStmt& n) {
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
    // This code is required since it is possible for n.iftrue to be a nullptr.
    // It needs to stay in place because its possible for verbose logging to print
    // pre-simplified code; this would cause a null pointer to be dereferencd and hence a crash.
    emit("if !(");
    n.cond->Accept(*this);
    emit(")\n");
    n.iffalse->Accept(*this);
  }
}

void EmitC::Visit(const sem::ForStmt& n) {
  emitTab();
  emit("for (int ");
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

void EmitC::Visit(const sem::WhileStmt& n) {
  emitTab();
  emit("while (");
  n.cond->Accept(*this);
  emit(")\n");
  n.inner->Accept(*this);
}

void EmitC::Visit(const sem::BarrierStmt& n) { throw std::runtime_error("Barrier unimplemented in EmitC"); }

void EmitC::Visit(const sem::ReturnStmt& n) {
  emitTab();
  emit("return");
  if (n.value) {
    emit(" (");
    n.value->Accept(*this);
    emit(")");
  }
  emit(";\n");
}

void EmitC::Visit(const sem::Function& n) {
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

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
