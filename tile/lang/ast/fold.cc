// Copyright 2019 Intel Corporation.

#include "tile/lang/ast/fold.h"

namespace vertexai {
namespace tile {
namespace lang {
namespace ast {

namespace {

ExprPtr make_number(ExprPtr, int64_t value) { return std::make_shared<IntConst>(value); }
DimExprPtr make_number(DimExprPtr, int64_t value) { return std::make_shared<DimIntExpr>(value); }
PolyExprPtr make_number(PolyExprPtr, int64_t value) { return std::make_shared<PolyLiteral>(value); }

ExprPtr make_number(ExprPtr, double value) { return std::make_shared<FloatConst>(value); }

template <typename NumType1, typename NumType2, typename ExprType>
std::shared_ptr<ExprType> fold_add(        //
    const std::shared_ptr<ExprType>& lhs,  //
    const std::shared_ptr<ExprType>& rhs) {
  auto lhs_num = std::dynamic_pointer_cast<NumType1>(lhs);
  auto rhs_num = std::dynamic_pointer_cast<NumType2>(rhs);
  if (lhs_num && rhs_num) {
    return make_number(lhs, lhs_num->value + rhs_num->value);
  }
  if (lhs_num && lhs_num->value == 0) {
    return rhs;
  }
  if (rhs_num && rhs_num->value == 0) {
    return lhs;
  }
  return nullptr;
}

template <typename NumType1, typename NumType2, typename OpExprType, typename NegOpType, typename ExprType>
std::shared_ptr<ExprType> fold_sub(        //
    const std::shared_ptr<ExprType>& lhs,  //
    const std::shared_ptr<ExprType>& rhs,  //
    NegOpType neg_op) {
  auto lhs_num = std::dynamic_pointer_cast<NumType1>(lhs);
  auto rhs_num = std::dynamic_pointer_cast<NumType2>(rhs);
  if (lhs_num && rhs_num) {
    return make_number(lhs, lhs_num->value - rhs_num->value);
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

template <typename NumType1, typename NumType2, typename ExprType>
std::shared_ptr<ExprType> fold_mul(        //
    const std::shared_ptr<ExprType>& lhs,  //
    const std::shared_ptr<ExprType>& rhs) {
  auto lhs_num = std::dynamic_pointer_cast<NumType1>(lhs);
  auto rhs_num = std::dynamic_pointer_cast<NumType2>(rhs);
  if (lhs_num && rhs_num) {
    return make_number(lhs, lhs_num->value * rhs_num->value);
  }
  if (lhs_num && lhs_num->value == 1) {
    return rhs;
  }
  if (rhs_num && rhs_num->value == 1) {
    return lhs;
  }
  return nullptr;
}

template <typename NumType1, typename NumType2, typename ExprType>
std::shared_ptr<ExprType> fold_div(        //
    const std::shared_ptr<ExprType>& lhs,  //
    const std::shared_ptr<ExprType>& rhs) {
  auto lhs_num = std::dynamic_pointer_cast<NumType1>(lhs);
  auto rhs_num = std::dynamic_pointer_cast<NumType2>(rhs);
  if (lhs_num && rhs_num) {
    return make_number(lhs, lhs_num->value / rhs_num->value);
  }
  if (lhs_num && lhs_num->value == 0) {
    return lhs;
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
      auto ret = fold_add<IntType, IntType>(args[0], args[1]);
      if (ret) {
        return ret;
      }
    } break;
    case IntOp::Sub: {
      auto ret = fold_sub<IntType, IntType, OpType>(args[0], args[1], IntOp::Neg);
      if (ret) {
        return ret;
      }
    } break;
    case IntOp::Mul: {
      auto ret = fold_mul<IntType, IntType>(args[0], args[1]);
      if (ret) {
        return ret;
      }
    } break;
    case IntOp::Div: {
      auto ret = fold_div<IntType, IntType>(args[0], args[1]);
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

ExprPtr MakeGradOverride(const std::shared_ptr<ExprDerivEntry>& fn, const std::vector<ExprPtr>& ins,
                         const ExprPtr& out) {  //
  auto expr = std::make_shared<GradOverrideExpr>(fn, ins, out);
  expr->ComputeShape();
  return expr;
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
    if (auto ret = fold_add<IntConst, IntConst>(args[0], args[1])) {
      return ret;
    }
    if (auto ret = fold_add<FloatConst, FloatConst>(args[0], args[1])) {
      return ret;
    }
    if (auto ret = fold_add<IntConst, FloatConst>(args[0], args[1])) {
      return ret;
    }
    if (auto ret = fold_add<FloatConst, IntConst>(args[0], args[1])) {
      return ret;
    }
  } else if (fn == "sub") {
    if (auto ret = fold_sub<IntConst, IntConst, CallExpr>(args[0], args[1], "neg")) {
      return ret;
    }
    if (auto ret = fold_sub<FloatConst, FloatConst, CallExpr>(args[0], args[1], "neg")) {
      return ret;
    }
    if (auto ret = fold_sub<IntConst, FloatConst, CallExpr>(args[0], args[1], "neg")) {
      return ret;
    }
    if (auto ret = fold_sub<FloatConst, IntConst, CallExpr>(args[0], args[1], "neg")) {
      return ret;
    }
  } else if (fn == "mul") {
    if (auto ret = fold_mul<IntConst, IntConst>(args[0], args[1])) {
      return ret;
    }
    if (auto ret = fold_mul<FloatConst, FloatConst>(args[0], args[1])) {
      return ret;
    }
    if (auto ret = fold_mul<IntConst, FloatConst>(args[0], args[1])) {
      return ret;
    }
    if (auto ret = fold_mul<FloatConst, IntConst>(args[0], args[1])) {
      return ret;
    }
  } else if (fn == "div") {
    if (auto ret = fold_div<IntConst, IntConst>(args[0], args[1])) {
      return ret;
    }
    if (auto ret = fold_div<FloatConst, FloatConst>(args[0], args[1])) {
      return ret;
    }
    if (auto ret = fold_div<IntConst, FloatConst>(args[0], args[1])) {
      return ret;
    }
    if (auto ret = fold_div<FloatConst, IntConst>(args[0], args[1])) {
      return ret;
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
