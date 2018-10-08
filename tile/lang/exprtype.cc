// Copyright 2017-2018 Intel Corporation.

#include "tile/lang/exprtype.h"

namespace vertexai {
namespace tile {
namespace lang {
namespace {

// Returns the Plaid arithmetic conversion rank of a type.
unsigned Rank(sem::Type ty) {
  if (ty.base != sem::Type::VALUE && ty.base != sem::Type::INDEX) {
    return 1;
  }
  auto dtype = ty.dtype;
  if (ty.base == sem::Type::INDEX) {
    dtype = DataType::INT32;
  }
  switch (dtype) {
    case DataType::BOOLEAN:
      return 2;
    case DataType::INT8:
      return 3;
    case DataType::UINT8:
      return 4;
    case DataType::INT16:
      return 5;
    case DataType::UINT16:
      return 6;
    case DataType::INT32:
      return 7;
    case DataType::UINT32:
      return 8;
    case DataType::INT64:
      return 9;
    case DataType::UINT64:
      return 10;
    case DataType::FLOAT16:
      return 11;
    case DataType::FLOAT32:
      return 12;
    case DataType::FLOAT64:
      return 13;
    default:
      throw std::logic_error{"Invalid type found in typecheck"};
  }
}

}  // namespace

sem::Type ExprType::TypeOf(const lang::Scope<sem::Type>* scope, bool enable_fp16, const sem::ExprPtr& expr) {
  ExprType et{scope, enable_fp16};
  expr->Accept(et);
  return et.ty_;
}

sem::Type ExprType::TypeOf(const lang::Scope<sem::Type>* scope, bool enable_fp16, const sem::LValPtr& lvalue) {
  ExprType et{scope, enable_fp16};
  lvalue->Accept(et);
  return et.ty_;
}

sem::Type Promote(const std::vector<sem::Type>& types) {
  sem::Type result{sem::Type::VALUE};
  unsigned rank_so_far = 0;
  for (auto ty : types) {
    switch (ty.base) {
      case sem::Type::INDEX:
      case sem::Type::VALUE:
        break;
      case sem::Type::POINTER_MUT:
      case sem::Type::POINTER_CONST:
        // Promotion of pointer types only happens when we're using a
        // pointer in a binary op to compute an address.  The result
        // is always a pointer type.
        return ty;
      default:
        throw std::logic_error{"Void type found during typecheck promotion"};
    }
    if (result.vec_width < ty.vec_width) {
      result.vec_width = ty.vec_width;
    }
    auto rank = Rank(ty);
    if (rank_so_far < rank) {
      rank_so_far = rank;
      result.dtype = ty.dtype;
      result.base = ty.base;
    }
  }
  return result;
}

void ExprType::Visit(const sem::IntConst& n) {
  DataType dtype;

  // This isn't quite correct; when we see an integer constant, we
  // ought to propagate its type lazily, so that it doesn't impose
  // constraints on expressions that use it (ala Go).  On the other
  // hand, this works pretty well in most cases, and it has the
  // advantage of being simple.
  if (n.value < 0) {
    if ((-127 - 1) /* OpenCL SCHAR_MIN */ <= n.value) {
      dtype = DataType::INT8;
    } else if ((-32767 - 1) /* OpenCL SHRT_MIN */ <= n.value) {
      dtype = DataType::INT16;
    } else if ((-2147483647 - 1) /* OpenCL INT_MIN */ <= n.value) {
      dtype = DataType::INT32;
    } else {
      dtype = DataType::INT64;
    }
  } else {
    if (n.value <= 127 /* OpenCL SCHAR_MAX */) {
      dtype = DataType::INT8;
    } else if (n.value <= 32767 /* OpenCL SHRT_MAX */) {
      dtype = DataType::INT16;
    } else if (n.value <= 2147483647 /* OpenCL INT_MAX */) {
      dtype = DataType::INT32;
    } else {
      dtype = DataType::INT64;
    }
  }

  ty_ = sem::Type{sem::Type::VALUE, dtype};
  IVLOG(5, "ExprType(IntConst): " << ty_);
}

void ExprType::Visit(const sem::FloatConst& n) {
  // This definitely isn't correct; when we see a float constant, we
  // ought to propagate its type lazily, so that it doesn't impose
  // contraints on expressions that use it (ala Go).
  ty_ = sem::Type{sem::Type::VALUE, DataType::FLOAT32};
  IVLOG(5, "ExprType(FloatConst): " << ty_);
}

void ExprType::Visit(const sem::LookupLVal& n) {
  auto result = scope_->Lookup(n.name);
  if (!result) {
    throw std::out_of_range{"Undeclared reference: " + n.name};
  }
  ty_ = *result;
  IVLOG(5, "ExprType(LookupLVal[" << n.name << "]): " << ty_);
}

void ExprType::Visit(const sem::LoadExpr& n) {
  n.inner->Accept(*this);
  if (ty_.dtype == DataType::FLOAT16 && !enable_fp16_) {
    // No fp16 support => automatically promote loads to 32-bit.
    ty_.dtype = DataType::FLOAT32;
  }
  IVLOG(5, "ExprType(LoadExpr): " << ty_);
}

void ExprType::Visit(const sem::StoreStmt&) { throw std::logic_error{"Unexpected expression component"}; }

void ExprType::Visit(const sem::SubscriptLVal& n) {
  n.ptr->Accept(*this);
  ty_.base = sem::Type::VALUE;
  ty_.array = 0;
  ty_.region = sem::Type::NORMAL;
  IVLOG(5, "ExprType(SubscriptLVal): " << ty_);
}

void ExprType::Visit(const sem::DeclareStmt&) { throw std::logic_error{"Unexpected expression component"}; }

void ExprType::Visit(const sem::UnaryExpr& n) {
  n.inner->Accept(*this);
  if (n.op == "!") {
    AdjustLogicOpResult();
  } else if (n.op == "*") {
    if (ty_.base != sem::Type::POINTER_MUT && ty_.base != sem::Type::POINTER_CONST) {
      throw std::logic_error{"Dereferencing a non-pointer in typecheck"};
    }
    ty_.base = sem::Type::VALUE;
  } else if (n.op == "&") {
    if (ty_.base != sem::Type::VALUE) {
      throw std::logic_error{"Taking the address of a non-value in typecheck"};
    }
    ty_.base = sem::Type::POINTER_MUT;
  } else if (n.op != "++" && n.op != "--" && n.op != "-" && n.op != "+") {
    throw std::logic_error{"Unrecognized unary operation in typecheck: " + n.op};
  }
  IVLOG(5, "ExprType(UnaryExpr[" << n.op << "]): " << ty_);
}

void ExprType::Visit(const sem::BinaryExpr& n) {
  ty_ = Promote({TypeOf(n.lhs), TypeOf(n.rhs)});
  if (n.op == ">" || n.op == ">=" || n.op == "<" || n.op == "<=" || n.op == "==" || n.op == "!=" || n.op == "&&" ||
      n.op == "||") {
    AdjustLogicOpResult();
  }
  IVLOG(5, "ExprType(BinaryExpr[" << n.op << "]): " << ty_);
}

void ExprType::Visit(const sem::CondExpr& n) {
  ty_ = Promote({TypeOf(n.tcase), TypeOf(n.fcase)});
  ty_.vec_width = std::max(ty_.vec_width, TypeOf(n.cond).vec_width);
  IVLOG(5, "ExprType(CondExpr): " << ty_);
}

void ExprType::Visit(const sem::SelectExpr& n) {
  ty_ = Promote({TypeOf(n.tcase), TypeOf(n.fcase)});
  ty_.vec_width = std::max(ty_.vec_width, TypeOf(n.cond).vec_width);
  IVLOG(5, "ExprType(SelectExpr): " << ty_);
}

void ExprType::Visit(const sem::ClampExpr& n) {
  n.val->Accept(*this);
  IVLOG(5, "ExprType(ClampExpr): " << ty_);
}

void ExprType::Visit(const sem::CastExpr& n) {
  // TODO: Consider using the type of the cast expression.  Currently,
  // the compiler is inserting casts that work but are technically
  // incorrect for OpenCL; those should perhaps be removed so that
  // only actually-semantically-useful casts remain, allowing this
  // code to insert those casts.
  n.val->Accept(*this);
  IVLOG(5, "ExprType(CastExpr): " << ty_);
}

void ExprType::Visit(const sem::CallExpr& n) {
  std::vector<sem::Type> types;
  for (auto val : n.vals) {
    types.push_back(TypeOf(val));
  }
  ty_ = Promote(types);
  IVLOG(5, "ExprType(CallExpr): " << ty_);
}

void ExprType::Visit(const sem::LimitConst& n) {
  ty_.base = sem::Type::INDEX;
  ty_.dtype = n.type;
  IVLOG(5, "ExprType(LimitConst): " << ty_);
}

void ExprType::Visit(const sem::IndexExpr& n) {
  ty_.base = sem::Type::INDEX;
  IVLOG(5, "ExprType(IndexExpr): " << ty_);
}

void ExprType::Visit(const sem::Block&) { throw std::logic_error{"Unexpected expression component"}; }

void ExprType::Visit(const sem::IfStmt&) { throw std::logic_error{"Unexpected expression component"}; }

void ExprType::Visit(const sem::ForStmt&) { throw std::logic_error{"Unexpected expression component"}; }

void ExprType::Visit(const sem::WhileStmt&) { throw std::logic_error{"Unexpected expression component"}; }

void ExprType::Visit(const sem::BarrierStmt&) { throw std::logic_error{"Unexpected expression component"}; }

void ExprType::Visit(const sem::ReturnStmt&) { throw std::logic_error{"Unexpected expression component"}; }

void ExprType::Visit(const sem::Function&) { throw std::logic_error{"Unexpected expression component"}; }

ExprType::ExprType(const lang::Scope<sem::Type>* scope, bool enable_fp16) : scope_{scope}, enable_fp16_{enable_fp16} {}

sem::Type ExprType::TypeOf(const sem::ExprPtr& expr) {
  ExprType et{scope_, enable_fp16_};
  expr->Accept(et);
  return et.ty_;
}

void ExprType::AdjustLogicOpResult() {
  ty_.base = sem::Type::VALUE;
  if (ty_.vec_width == 1) {
    ty_.dtype = DataType::INT32;
  } else {
    switch (ty_.dtype) {
      case DataType::BOOLEAN:
        throw std::logic_error{"Invalid boolean vector type found in typecheck"};
      case DataType::INT8:
      case DataType::UINT8:
        ty_.dtype = DataType::INT8;
        break;
      case DataType::INT16:
      case DataType::UINT16:
      case DataType::FLOAT16:
        ty_.dtype = DataType::INT16;
        break;
      case DataType::INT32:
      case DataType::UINT32:
      case DataType::FLOAT32:
        ty_.dtype = DataType::INT32;
        break;
      case DataType::INT64:
      case DataType::UINT64:
      case DataType::FLOAT64:
        ty_.dtype = DataType::INT64;
        break;
      default:
        throw std::logic_error{"Invalid vector type found in typecheck"};
    }
  }
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
