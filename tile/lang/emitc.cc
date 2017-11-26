
#include "tile/lang/emitc.h"

#include "tile/lang/fpconv.h"

namespace vertexai {
namespace tile {
namespace sem {

inline std::string c_dtype(const lang::DataType& dt) {
  std::string base;
  switch (dt) {
    case lang::DataType::BOOLEAN:
      base = "bool";
      break;
    case lang::DataType::INT8:
      base = "char";
      break;
    case lang::DataType::INT16:
      base = "short";
      break;
    case lang::DataType::INT32:
      base = "int";
      break;
    case lang::DataType::INT64:
      base = "long";
      break;
    case lang::DataType::UINT8:
      base = "uchar";
      break;
    case lang::DataType::UINT16:
      base = "ushort";
      break;
    case lang::DataType::UINT32:
      base = "uint";
      break;
    case lang::DataType::UINT64:
      base = "ulong";
      break;
    case lang::DataType::FLOAT16:
      base = "half";
      break;
    case lang::DataType::FLOAT32:
      base = "float";
      break;
    case lang::DataType::FLOAT64:
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

void EmitC::operator()(const sem::IntConst& n) { emit(std::to_string(n.value)); }

void EmitC::operator()(const sem::FloatConst& n) {
  std::string c = lang::DoubleToString(n.value);
  if (c.find_first_of(".e") == std::string::npos) {
    c += ".0";
  }
  emit(c + "f");
}

void EmitC::operator()(const sem::LookupLVal& n) { emit(n.name); }

void EmitC::operator()(const sem::LoadExpr& n) { boost::apply_visitor(*this, *n.inner); }

void EmitC::operator()(const sem::StoreStmt& n) {
  emitTab();
  boost::apply_visitor(*this, *n.lhs);
  emit(" = ");
  boost::apply_visitor(*this, *n.rhs);
  emit(";\n");
}

void EmitC::operator()(const sem::SubscriptLVal& n) {
  boost::apply_visitor(*this, *n.ptr);
  emit("[");
  boost::apply_visitor(*this, *n.offset);
  emit("]");
}

void EmitC::operator()(const sem::DeclareStmt& n) {
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
        boost::apply_visitor(*this, *n.init);
        emit(", ");
      }
      emit("}");
    } else {
      boost::apply_visitor(*this, *n.init);
    }
  }
  emit(";\n");
}

void EmitC::operator()(const sem::UnaryExpr& n) {
  emit("(");
  emit(n.op);
  boost::apply_visitor(*this, *n.inner);
  emit(")");
}

void EmitC::operator()(const sem::BinaryExpr& n) {
  emit("(");
  boost::apply_visitor(*this, *n.lhs);
  emit(" ");
  emit(n.op);
  emit(" ");
  boost::apply_visitor(*this, *n.rhs);
  emit(")");
}

void EmitC::operator()(const sem::CondExpr& n) {
  emit("(");
  boost::apply_visitor(*this, *n.cond);
  emit(" ? ");
  boost::apply_visitor(*this, *n.tcase);
  emit(" : ");
  boost::apply_visitor(*this, *n.fcase);
  emit(")");
}

void EmitC::operator()(const sem::SelectExpr& n) {
  emit("select(");
  boost::apply_visitor(*this, *n.fcase);
  emit(", ");
  boost::apply_visitor(*this, *n.tcase);
  emit(", ");
  boost::apply_visitor(*this, *n.cond);
  emit(")");
}

void EmitC::operator()(const sem::ClampExpr& n) {
  emit("clamp(");
  boost::apply_visitor(*this, *n.val);
  emit(", ");
  boost::apply_visitor(*this, *n.min);
  emit(", ");
  boost::apply_visitor(*this, *n.max);
  emit(")");
}

void EmitC::operator()(const sem::CastExpr& n) {
  emit("((");
  emitType(n.type);
  emit(")");
  boost::apply_visitor(*this, *n.val);
  emit(")");
}

void EmitC::operator()(const sem::CallExpr& n) {
  boost::apply_visitor(*this, *n.func);
  emit("(");
  for (size_t i = 0; i < n.vals.size(); i++) {
    if (i) {
      emit(", ");
    }
    boost::apply_visitor(*this, *n.vals[i]);
  }
  emit(")");
}

static std::map<std::pair<lang::DataType, sem::LimitConst::Which>, std::string> LimitConstLookup = {
    {{lang::DataType::BOOLEAN, sem::LimitConst::MIN}, "0"},
    {{lang::DataType::INT8, sem::LimitConst::MIN}, "SCHAR_MIN"},
    {{lang::DataType::INT16, sem::LimitConst::MIN}, "SHRT_MIN"},
    {{lang::DataType::INT32, sem::LimitConst::MIN}, "INT_MIN"},
    {{lang::DataType::INT64, sem::LimitConst::MIN}, "LONG_MIN"},
    {{lang::DataType::UINT8, sem::LimitConst::MIN}, "0"},
    {{lang::DataType::UINT16, sem::LimitConst::MIN}, "0"},
    {{lang::DataType::UINT32, sem::LimitConst::MIN}, "0"},
    {{lang::DataType::UINT64, sem::LimitConst::MIN}, "0"},
    {{lang::DataType::FLOAT16, sem::LimitConst::MIN}, "-65504"},
    {{lang::DataType::FLOAT32, sem::LimitConst::MIN}, "-FLT_MAX"},
    {{lang::DataType::FLOAT64, sem::LimitConst::MIN}, "-DBL_MAX"},

    {{lang::DataType::BOOLEAN, sem::LimitConst::MAX}, "0"},
    {{lang::DataType::INT8, sem::LimitConst::MAX}, "SCHAR_MAX"},
    {{lang::DataType::INT16, sem::LimitConst::MAX}, "SHRT_MAX"},
    {{lang::DataType::INT32, sem::LimitConst::MAX}, "INT_MAX"},
    {{lang::DataType::INT64, sem::LimitConst::MAX}, "LONG_MAX"},
    {{lang::DataType::UINT8, sem::LimitConst::MAX}, "UCHAR_MAX"},
    {{lang::DataType::UINT16, sem::LimitConst::MAX}, "USHRT_MAX"},
    {{lang::DataType::UINT32, sem::LimitConst::MAX}, "UINT_MAX"},
    {{lang::DataType::UINT64, sem::LimitConst::MAX}, "ULONG_MAX"},
    {{lang::DataType::FLOAT16, sem::LimitConst::MAX}, "65504"},
    {{lang::DataType::FLOAT32, sem::LimitConst::MAX}, "FLT_MAX"},
    {{lang::DataType::FLOAT64, sem::LimitConst::MAX}, "DBL_MAX"},
};

void EmitC::operator()(const sem::LimitConst& n) {
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

void EmitC::operator()(const sem::IndexExpr& n) { throw std::runtime_error("IndexExpr unimplemented in EmitC"); }

void EmitC::operator()(const sem::Block& n) {
  emitTab();
  emit("{\n");
  ++indent_;
  for (const sem::StmtPtr& ptr : *n.statements) {
    boost::apply_visitor(*this, *ptr);
  }
  --indent_;
  emitTab();
  emit("}\n");
}

void EmitC::operator()(const sem::IfStmt& n) {
  emitTab();
  // TODO: Re-combine these print cases; handle the only-iffalse case in simplification.
  if (n.iftrue && n.iffalse) {
    emit("if (");
    boost::apply_visitor(*this, *n.cond);
    emit(")\n");
    (*this)(*n.iftrue);
    emitTab();
    emit("else\n");
    (*this)(*n.iffalse);
  } else if (n.iftrue) {
    emit("if (");
    boost::apply_visitor(*this, *n.cond);
    emit(")\n");
    (*this)(*n.iftrue);
  } else if (n.iffalse) {
    emit("if (!");
    boost::apply_visitor(*this, *n.cond);
    emit(")\n");
    (*this)(*n.iffalse);
  }
}

void EmitC::operator()(const sem::ForStmt& n) {
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
  (*this)(*n.inner);
}

void EmitC::operator()(const sem::WhileStmt& n) {
  emitTab();
  emit("while (");
  boost::apply_visitor(*this, *n.cond);
  emit(")\n");
  (*this)(*n.inner);
}

void EmitC::operator()(const sem::BarrierStmt& n) { throw std::runtime_error("Barrier unimplemented in EmitC"); }

void EmitC::operator()(const sem::ReturnStmt& n) {
  emitTab();
  emit("return");
  if (n.value) {
    emit(" (");
    boost::apply_visitor(*this, *n.value);
    emit(")");
  }
  emit(";\n");
}

void EmitC::operator()(const sem::Function& n) {
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
  (*this)(*n.body);
}

void Print::operator()(const sem::CallExpr& n) {
  boost::apply_visitor(*static_cast<EmitC*>(this), *n.func);
  emit("(");
  for (size_t i = 0; i < n.vals.size(); i++) {
    boost::apply_visitor(*static_cast<EmitC*>(this), *n.vals[i]);
    if (i != n.vals.size() - 1) {
      emit(", ");
    }
  }
  emit(")");
}

void Print::operator()(const sem::IndexExpr& n) {
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

void Print::operator()(const sem::BarrierStmt& n) {
  emitTab();
  emit("barrier();\n");
}

void Print::operator()(const sem::Function& f) {
  emit("kernel ");
  EmitC::operator()(f);
}

}  // namespace sem
}  // namespace tile
}  // namespace vertexai
