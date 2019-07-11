// Copyright 2019 Intel Corporation.

#include "tile/lang/ast/fold.h"

namespace vertexai {
namespace tile {
namespace lang {
namespace ast {

namespace {

template <typename NumType, typename ExprType>
std::shared_ptr<ExprType> fold_add(        //
    const std::shared_ptr<ExprType>& lhs,  //
    const std::shared_ptr<ExprType>& rhs) {
  auto lhs_num = std::dynamic_pointer_cast<NumType>(lhs);
  auto rhs_num = std::dynamic_pointer_cast<NumType>(rhs);
  if (lhs_num && rhs_num) {
    return std::make_shared<NumType>(lhs_num->value + rhs_num->value);
  }
  if (lhs_num && lhs_num->value == 0) {
    return rhs;
  }
  if (rhs_num && rhs_num->value == 0) {
    return lhs;
  }
  return nullptr;
}

template <typename NumType, typename OpExprType, typename NegOpType, typename ExprType>
std::shared_ptr<ExprType> fold_sub(        //
    const std::shared_ptr<ExprType>& lhs,  //
    const std::shared_ptr<ExprType>& rhs,  //
    NegOpType neg_op) {
  auto lhs_num = std::dynamic_pointer_cast<NumType>(lhs);
  auto rhs_num = std::dynamic_pointer_cast<NumType>(rhs);
  if (lhs_num && rhs_num) {
    return std::make_shared<NumType>(lhs_num->value - rhs_num->value);
  }
  // TODO: deal with ComputeShape
  // if (lhs_num && lhs_num->value == 0) {
  //   std::vector<std::shared_ptr<ExprType>> args{rhs};
  //   return std::make_shared<OpExprType>(neg_op, args);
  // }
  if (rhs_num && rhs_num->value == 0) {
    return lhs;
  }
  return nullptr;
}

template <typename NumType, typename ExprType>
std::shared_ptr<ExprType> fold_mul(        //
    const std::shared_ptr<ExprType>& lhs,  //
    const std::shared_ptr<ExprType>& rhs) {
  auto lhs_num = std::dynamic_pointer_cast<NumType>(lhs);
  auto rhs_num = std::dynamic_pointer_cast<NumType>(rhs);
  if (lhs_num && rhs_num) {
    return std::make_shared<NumType>(lhs_num->value * rhs_num->value);
  }
  if (lhs_num && lhs_num->value == 1) {
    return rhs;
  }
  if (rhs_num && rhs_num->value == 1) {
    return lhs;
  }
  return nullptr;
}

template <typename NumType, typename ExprType>
std::shared_ptr<ExprType> fold_div(        //
    const std::shared_ptr<ExprType>& lhs,  //
    const std::shared_ptr<ExprType>& rhs) {
  auto lhs_num = std::dynamic_pointer_cast<NumType>(lhs);
  auto rhs_num = std::dynamic_pointer_cast<NumType>(rhs);
  if (lhs_num && rhs_num) {
    return std::make_shared<NumType>(lhs_num->value / rhs_num->value);
  }
  if (lhs_num && lhs_num->value == 0) {
    return std::make_shared<NumType>(0);
  }
  if (rhs_num && rhs_num->value == 1) {
    return lhs;
  }
  return nullptr;
}

template <typename IntType, typename OpType, typename ExprType>
std::shared_ptr<ExprType> MakeOp(IntOp op, const std::vector<std::shared_ptr<ExprType>>& args) {
  switch (op) {
    case IntOp::Neg: {
      auto int_expr = std::dynamic_pointer_cast<IntType>(args[0]);
      if (int_expr) {
        return std::make_shared<IntType>(-int_expr->value);
      }
    } break;
    case IntOp::Add: {
      auto ret = fold_add<IntType>(args[0], args[1]);
      if (ret) {
        return ret;
      }
    } break;
    case IntOp::Sub: {
      auto ret = fold_sub<IntType, OpType>(args[0], args[1], IntOp::Neg);
      if (ret) {
        return ret;
      }
    } break;
    case IntOp::Mul: {
      auto ret = fold_mul<IntType>(args[0], args[1]);
      if (ret) {
        return ret;
      }
    } break;
    case IntOp::Div: {
      auto ret = fold_div<IntType>(args[0], args[1]);
      if (ret) {
        return ret;
      }
    } break;
  }
  return std::make_shared<OpType>(op, args);
}

}  // namespace

PolyExprPtr MakeOp(IntOp op, const std::vector<PolyExprPtr>& args) {  //
  return MakeOp<PolyLiteral, PolyOpExpr>(op, args);
}

DimExprPtr MakeOp(IntOp op, const std::vector<DimExprPtr>& args) {  //
  return MakeOp<DimIntExpr, DimOpExpr>(op, args);
}

ExprPtr MakeCall(const std::string& fn, const std::vector<ExprPtr>& args) {
  if (fn == "neg") {
    auto int_expr = std::dynamic_pointer_cast<IntConst>(args[0]);
    if (int_expr) {
      return std::make_shared<IntConst>(-int_expr->value);
    }
    auto float_expr = std::dynamic_pointer_cast<FloatConst>(args[0]);
    if (float_expr) {
      return std::make_shared<FloatConst>(-float_expr->value);
    }
  } else if (fn == "add") {
    auto int_ret = fold_add<IntConst>(args[0], args[1]);
    if (int_ret) {
      return int_ret;
    }
    auto float_ret = fold_add<FloatConst>(args[0], args[1]);
    if (float_ret) {
      return float_ret;
    }
  } else if (fn == "sub") {
    auto int_ret = fold_sub<IntConst, CallExpr>(args[0], args[1], "neg");
    if (int_ret) {
      return int_ret;
    }
    auto float_ret = fold_sub<FloatConst, CallExpr>(args[0], args[1], "neg");
    if (float_ret) {
      return float_ret;
    }
  } else if (fn == "mul") {
    auto int_ret = fold_mul<IntConst>(args[0], args[1]);
    if (int_ret) {
      return int_ret;
    }
    auto float_ret = fold_mul<FloatConst>(args[0], args[1]);
    if (float_ret) {
      return float_ret;
    }
  } else if (fn == "div") {
    auto int_ret = fold_div<IntConst>(args[0], args[1]);
    if (int_ret) {
      return int_ret;
    }
    auto float_ret = fold_div<FloatConst>(args[0], args[1]);
    if (float_ret) {
      return float_ret;
    }
  }
  auto expr = std::make_shared<CallExpr>(fn, args);
  expr->ComputeShape();
  return expr;
}

}  // namespace ast
}  // namespace lang
}  // namespace tile
}  // namespace vertexai
