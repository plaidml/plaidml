// Copyright 2017, Vertex.AI.

#include "tile/hal/opencl/exprtype.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {
namespace {

// Returns the Plaid arithmetic conversion rank of a type.
unsigned Rank(sem::Type ty) {
  if (ty.base != sem::Type::VALUE && ty.base != sem::Type::INDEX) {
    return 1;
  }
  auto dtype = ty.dtype;
  if (ty.base == sem::Type::INDEX) {
    dtype = lang::DataType::INT32;
  }
  switch (dtype) {
    case lang::DataType::BOOLEAN:
      return 2;
    case lang::DataType::INT8:
      return 3;
    case lang::DataType::UINT8:
      return 4;
    case lang::DataType::INT16:
      return 5;
    case lang::DataType::UINT16:
      return 6;
    case lang::DataType::INT32:
      return 7;
    case lang::DataType::UINT32:
      return 8;
    case lang::DataType::INT64:
      return 9;
    case lang::DataType::UINT64:
      return 10;
    case lang::DataType::FLOAT16:
      return 11;
    case lang::DataType::FLOAT32:
      return 12;
    case lang::DataType::FLOAT64:
      return 13;
    default:
      throw std::logic_error{"Invalid type found in typecheck"};
  }
}

}  // namespace

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

sem::Type ExprType::TypeOf(const lang::Scope<sem::Type>* scope, bool cl_khr_fp16, const sem::ExprPtr& expr) {
  ExprType et{scope, cl_khr_fp16};
  return boost::apply_visitor(et, *expr);
}

sem::Type ExprType::TypeOf(const lang::Scope<sem::Type>* scope, bool cl_khr_fp16, const sem::LValPtr& lvalue) {
  ExprType et{scope, cl_khr_fp16};
  return boost::apply_visitor(et, *lvalue);
}

sem::Type ExprType::AdjustLogicOpResult(sem::Type ty) {
  ty.base = sem::Type::VALUE;
  if (ty.vec_width == 1) {
    ty.dtype = lang::DataType::INT32;
  } else {
    switch (ty.dtype) {
      case lang::DataType::BOOLEAN:
        throw std::logic_error{"Invalid boolean vector type found in typecheck"};
      case lang::DataType::INT8:
      case lang::DataType::UINT8:
        ty.dtype = lang::DataType::INT8;
        break;
      case lang::DataType::INT16:
      case lang::DataType::UINT16:
      case lang::DataType::FLOAT16:
        ty.dtype = lang::DataType::INT16;
        break;
      case lang::DataType::INT32:
      case lang::DataType::UINT32:
      case lang::DataType::FLOAT32:
        ty.dtype = lang::DataType::INT32;
        break;
      case lang::DataType::INT64:
      case lang::DataType::UINT64:
      case lang::DataType::FLOAT64:
        ty.dtype = lang::DataType::INT64;
        break;
      default:
        throw std::logic_error{"Invalid vector type found in typecheck"};
    }
  }
  return ty;
}

sem::Type ExprType::operator()(const sem::IntConst& n) {
  lang::DataType dtype;

  // This isn't quite correct; when we see an integer constant, we
  // ought to propagate its type lazily, so that it doesn't impose
  // constraints on expressions that use it (ala Go).  On the other
  // hand, this works pretty well in most cases, and it has the
  // advantage of being simple.
  if (n.value < 0) {
    if ((-127 - 1) /* OpenCL SCHAR_MIN */ <= n.value) {
      dtype = lang::DataType::INT8;
    } else if ((-32767 - 1) /* OpenCL SHRT_MIN */ <= n.value) {
      dtype = lang::DataType::INT16;
    } else if ((-2147483647 - 1) /* OpenCL INT_MIN */ <= n.value) {
      dtype = lang::DataType::INT32;
    } else {
      dtype = lang::DataType::INT64;
    }
  } else {
    if (n.value <= 127 /* OpenCL SCHAR_MAX */) {
      dtype = lang::DataType::INT8;
    } else if (n.value <= 32767 /* OpenCL SHRT_MAX */) {
      dtype = lang::DataType::INT16;
    } else if (n.value <= 2147483647 /* OpenCL INT_MAX */) {
      dtype = lang::DataType::INT32;
    } else {
      dtype = lang::DataType::INT64;
    }
  }

  sem::Type ty{sem::Type::VALUE, dtype};
  IVLOG(5, "ExprType(IntConst): " << ty);
  return ty;
}

sem::Type ExprType::operator()(const sem::FloatConst& n) {
  // This definitely isn't correct; when we see a float constant, we
  // ought to propagate its type lazily, so that it doesn't impose
  // contraints on expressions that use it (ala Go).
  sem::Type ty{sem::Type::VALUE, lang::DataType::FLOAT32};
  IVLOG(5, "ExprType(FloatConst): " << ty);
  return ty;
}

sem::Type ExprType::operator()(const sem::LookupLVal& n) {
  auto result = scope_->Lookup(n.name);
  if (!result) {
    throw std::out_of_range{"Undeclared reference: " + n.name};
  }
  auto ty = *result;
  IVLOG(5, "ExprType(LookupLVal[" << n.name << "]): " << ty);
  return ty;
}

sem::Type ExprType::operator()(const sem::LoadExpr& n) {
  auto ty = boost::apply_visitor(*this, *n.inner);
  if (ty.dtype == lang::DataType::FLOAT16 && !cl_khr_fp16_) {
    // No fp16 support => automatically promote loads to 32-bit.
    ty.dtype = lang::DataType::FLOAT32;
  }
  IVLOG(5, "ExprType(LoadExpr): " << ty);
  return ty;
}

sem::Type ExprType::operator()(const sem::SubscriptLVal& n) {
  auto ty = boost::apply_visitor(*this, *n.ptr);
  ty.base = sem::Type::VALUE;
  ty.array = 0;
  ty.region = sem::Type::NORMAL;
  IVLOG(5, "ExprType(SubscriptLVal): " << ty);
  return ty;
}

sem::Type ExprType::operator()(const sem::UnaryExpr& n) {
  auto ty = boost::apply_visitor(*this, *n.inner);
  if (n.op == "!") {
    ty = AdjustLogicOpResult(ty);
  } else if (n.op == "*") {
    if (ty.base != sem::Type::POINTER_MUT && ty.base != sem::Type::POINTER_CONST) {
      throw std::logic_error{"Dereferencing a non-pointer in typecheck"};
    }
    ty.base = sem::Type::VALUE;
  } else if (n.op == "&") {
    if (ty.base != sem::Type::VALUE) {
      throw std::logic_error{"Taking the address of a non-value in typecheck"};
    }
    ty.base = sem::Type::POINTER_MUT;
  } else if (n.op != "++" && n.op != "--" && n.op != "-" && n.op != "+") {
    throw std::logic_error{"Unrecognized unary operation in typecheck: " + n.op};
  }
  IVLOG(5, "ExprType(UnaryExpr[" << n.op << "]): " << ty);
  return ty;
}

sem::Type ExprType::operator()(const sem::BinaryExpr& n) {
  auto ty = Promote({boost::apply_visitor(*this, *n.lhs), boost::apply_visitor(*this, *n.rhs)});
  if (n.op == ">" || n.op == ">=" || n.op == "<" || n.op == "<=" || n.op == "==" || n.op == "!=" || n.op == "&&" ||
      n.op == "||") {
    ty = AdjustLogicOpResult(ty);
  }
  IVLOG(5, "ExprType(BinaryExpr[" << n.op << "]): " << ty);
  return ty;
}

sem::Type ExprType::operator()(const sem::CondExpr& n) {
  auto ty = Promote({boost::apply_visitor(*this, *n.tcase), boost::apply_visitor(*this, *n.fcase)});
  IVLOG(5, "ExprType(CondExpr): " << ty);
  return ty;
}

sem::Type ExprType::operator()(const sem::SelectExpr& n) {
  auto ty = Promote({boost::apply_visitor(*this, *n.tcase), boost::apply_visitor(*this, *n.fcase)});
  IVLOG(5, "ExprType(SelectExpr): " << ty);
  return ty;
}

sem::Type ExprType::operator()(const sem::ClampExpr& n) {
  auto ty = boost::apply_visitor(*this, *n.val);
  IVLOG(5, "ExprType(ClampExpr): " << ty);
  return ty;
}

sem::Type ExprType::operator()(const sem::CastExpr& n) {
  // TODO: Consider using the type of the cast expression.  Currently,
  // the compiler is inserting casts that work but are technically
  // incorrect for OpenCL; those should perhaps be removed so that
  // only actually-semantically-useful casts remain, allowing this
  // code to insert those casts.
  auto ty = boost::apply_visitor(*this, *n.val);
  IVLOG(5, "ExprType(CastExpr): " << ty);
  return ty;
}

sem::Type ExprType::operator()(const sem::CallExpr& n) {
  std::vector<sem::Type> types;
  for (auto val : n.vals) {
    types.push_back(boost::apply_visitor(*this, *val));
  }
  auto ty = Promote(types);
  IVLOG(5, "ExprType(CallExpr): " << ty);
  return ty;
}

sem::Type ExprType::operator()(const sem::LimitConst& n) {
  sem::Type ty{sem::Type::INDEX};
  ty.dtype = n.type;
  IVLOG(5, "ExprType(LimitConst): " << ty);
  return ty;
}

sem::Type ExprType::operator()(const sem::IndexExpr& n) {
  sem::Type ty{sem::Type::INDEX};
  IVLOG(5, "ExprType(IndexExpr): " << ty);
  return ty;
}

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
