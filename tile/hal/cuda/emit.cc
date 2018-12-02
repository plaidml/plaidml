// Copyright 2018, Intel Corporation.

#include "tile/hal/cuda/emit.h"

#include <map>
#include <set>
#include <utility>

#include "tile/lang/exprtype.h"
#include "tile/lang/fpconv.h"
#include "tile/lang/scope.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cuda {

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
      return "long long";
    case DataType::UINT8:
      return "unsigned char";
    case DataType::UINT16:
      return "unsigned short";
    case DataType::UINT32:
      return "unsigned int";
    case DataType::FLOAT16:
      return "half";
    case DataType::FLOAT32:
      return "float";
    case DataType::UINT64:
      return "unsigned long long";
    case DataType::FLOAT64:
      return "double";
    default:
      throw std::runtime_error("Invalid tile type");
  }
}

static std::map<std::pair<DataType, sem::LimitConst::Which>, std::string> LimitConstLookup = {
    {{DataType::BOOLEAN, sem::LimitConst::MIN}, "0"},
    {{DataType::INT8, sem::LimitConst::MIN}, "-127"},
    {{DataType::INT16, sem::LimitConst::MIN}, "-32767"},
    {{DataType::INT32, sem::LimitConst::MIN}, "-2147483647"},
    {{DataType::INT64, sem::LimitConst::MIN}, "-4611686018427387902"},  // FIXME
    {{DataType::UINT8, sem::LimitConst::MIN}, "0"},
    {{DataType::UINT16, sem::LimitConst::MIN}, "0"},
    {{DataType::UINT32, sem::LimitConst::MIN}, "0"},
    {{DataType::UINT64, sem::LimitConst::MIN}, "0"},
    {{DataType::FLOAT16, sem::LimitConst::MIN}, "-65504"},
    {{DataType::FLOAT32, sem::LimitConst::MIN}, "-3.402823466e+38"},
    {{DataType::FLOAT64, sem::LimitConst::MIN}, "-1.7976931348623158e+308"},

    {{DataType::BOOLEAN, sem::LimitConst::MAX}, "1"},
    {{DataType::INT8, sem::LimitConst::MAX}, "127"},
    {{DataType::INT16, sem::LimitConst::MAX}, "32767"},
    {{DataType::INT32, sem::LimitConst::MAX}, "2147483647"},
    {{DataType::INT64, sem::LimitConst::MAX}, "4611686018427387902"},  // FIXME
    {{DataType::UINT8, sem::LimitConst::MAX}, "255"},
    {{DataType::UINT16, sem::LimitConst::MAX}, "65535"},
    {{DataType::UINT32, sem::LimitConst::MAX}, "4294967295"},
    {{DataType::UINT64, sem::LimitConst::MAX}, "4611686018427387902"},  // FIXME
    {{DataType::FLOAT16, sem::LimitConst::MAX}, "65504"},
    {{DataType::FLOAT32, sem::LimitConst::MAX}, "3.402823466e+38"},
    {{DataType::FLOAT64, sem::LimitConst::MAX}, "1.7976931348623158e+308"},
};

class Emitter : public sem::Visitor {
 public:
  explicit Emitter(std::ostringstream* oss) : oss_(*oss), scope_{nullptr} {}

  void Visit(const sem::IntConst& node) override { emit(std::to_string(node.value)); }

  void Visit(const sem::FloatConst& node) override {
    std::string c = lang::DoubleToString(node.value);
    if (c.find_first_of(".e") == std::string::npos) {
      c += ".0";
    }
    emit(c + "f");
  }

  void Visit(const sem::LookupLVal& node) override { emit(node.name); }

  void Visit(const sem::LoadExpr& node) override { node.inner->Accept(*this); }

  void Visit(const sem::StoreStmt& node) override {
    emitTab();
    node.lhs->Accept(*this);
    emit(" = ");
    node.rhs->Accept(*this);
    emit(";\n");
  }

  void Visit(const sem::SubscriptLVal& node) override {
    node.ptr->Accept(*this);
    emit("[");
    node.offset->Accept(*this);
    emit("]");
  }

  void Visit(const sem::DeclareStmt& node) override {
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
        node.init->Accept(*this);
      }
    }
    emit(";\n");

    scope_->Bind(node.name, node.type);
  }

  void Visit(const sem::UnaryExpr& node) override {
    emit("(");
    emit(node.op);
    node.inner->Accept(*this);
    emit(")");
  }

  void Visit(const sem::BinaryExpr& node) override {
    emit("(");
    node.lhs->Accept(*this);
    emit(" ");
    emit(node.op);
    emit(" ");
    node.rhs->Accept(*this);
    emit(")");
  }

  void Visit(const sem::CondExpr& node) override {
    emit("(");
    node.cond->Accept(*this);
    emit(" ? ");
    node.tcase->Accept(*this);
    emit(" : ");
    node.fcase->Accept(*this);
    emit(")");
  }

  void Visit(const sem::SelectExpr& node) override {
    emit("(");
    node.cond->Accept(*this);
    emit(" ? ");
    node.tcase->Accept(*this);
    emit(" : ");
    node.fcase->Accept(*this);
    emit(")");
  }

  void Visit(const sem::ClampExpr& node) override {
    emit("clamp(");
    node.val->Accept(*this);
    emit(", ");
    node.min->Accept(*this);
    emit(", ");
    node.max->Accept(*this);
    emit(")");
  }

  void Visit(const sem::CastExpr& node) override {
    emit("(");
    emitType(node.type);
    emit(")");
    node.val->Accept(*this);
  }

  void Visit(const sem::CallExpr& node) override {
    emit(node.name);
    emit("(");
    for (size_t i = 0; i < node.vals.size(); i++) {
      if (i) {
        emit(", ");
      }
      node.vals[i]->Accept(*this);
    }
    emit(")");
  }

  void Visit(const sem::LimitConst& node) override {
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

  void Visit(const sem::IndexExpr& node) override {
    const char* dims[] = {"x", "y", "z"};
    switch (node.type) {
      case sem::IndexExpr::GLOBAL:
        emit(printstring("blockIdx.%s * blockDim.%s + threadIdx.%s", dims[node.dim], dims[node.dim], dims[node.dim]));
        break;
      case sem::IndexExpr::GROUP:
        emit(printstring("blockIdx.%s", dims[node.dim]));
        break;
      case sem::IndexExpr::LOCAL:
        emit(printstring("threadIdx.%s", dims[node.dim]));
        break;
      default:
        throw std::runtime_error("Invalid IndexExpr type");
    }
  }

  void Visit(const sem::Block& node) override {
    auto previous_scope = scope_;
    lang::Scope<sem::Type> scope{scope_};
    scope_ = &scope;

    emitTab();
    emit("{\n");
    ++indent_;
    for (const sem::StmtPtr& ptr : node.statements) {
      ptr->Accept(*this);
    }
    --indent_;
    emitTab();
    emit("}\n");

    scope_ = previous_scope;
  }

  void Visit(const sem::IfStmt& node) override {
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
      // pre-simplified code; this would cause a null pointer to be dereferenced and hence a crash.
      emit("if !(");
      node.cond->Accept(*this);
      emit(")\n");
      node.iffalse->Accept(*this);
    }
  }

  void Visit(const sem::ForStmt& node) override {
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

  void Visit(const sem::WhileStmt& node) override {
    emitTab();
    emit("while (");
    node.cond->Accept(*this);
    emit(")\n");
    node.inner->Accept(*this);
  }

  void Visit(const sem::BarrierStmt& node) override {
    emitTab();
    emit("__syncthreads();\n");
  }

  void Visit(const sem::ReturnStmt& node) override {
    emitTab();
    emit("return");
    if (node.value) {
      emit(" (");
      node.value->Accept(*this);
      emit(")");
    }
    emit(";\n");
  }

  void Visit(const sem::Function& node) override {
    lang::Scope<sem::Type> scope;
    scope_ = &scope;

    emit("extern \"C\" __global__\n");
    emitType(node.ret);
    emit(" ");
    emit(node.name);
    emit("(");
    bool first_param = true;
    for (const auto& p : node.params) {
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
    node.body->Accept(*this);

    scope_ = nullptr;
  }

 private:
  void emit(const std::string& str) { oss_ << str; }

  void emitTab() { oss_ << std::string(indent_ << 1, ' '); }

  void emitType(const sem::Type& type) {
    if (type.region == sem::Type::LOCAL) {
      emit("__shared__ ");
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

  sem::Type typeOf(const sem::ExprPtr& expr) { return lang::ExprType::TypeOf(scope_, false, expr); }

  sem::Type typeOf(const sem::LValPtr& lvalue) { return lang::ExprType::TypeOf(scope_, false, lvalue); }

  void emitCast(const sem::Type& to, const sem::ExprPtr& expr) {
    emitType(to);
    emit("(");
    expr->Accept(*this);
    emit(")");
  }

 private:
  std::ostringstream& oss_;
  size_t indent_ = 0;
  lang::Scope<sem::Type>* scope_;
};

std::string EmitClamp() {
  return R"(
template<typename T, typename U>
__device__ T clamp(T val, U min, U max) {
  const T tmp = val < min ? min : val;
  return tmp > max ? max : tmp;
}
)";
}

std::string EmitCudaC(const std::vector<lang::KernelInfo>& kernels) {
  std::ostringstream src;

  src << EmitClamp() << "\n";

  std::set<std::string> seen;
  for (const auto& ki : kernels) {
    if (ki.ktype == lang::KernelType::kFunction) {
      if (!seen.count(ki.kname)) {
        Emitter emitter(&src);
        emitter.Visit(*ki.kfunc);
        src << "\n\n";
        seen.insert(ki.kname);
      }
    }
  }
  return src.str();
}

}  // namespace cuda
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
