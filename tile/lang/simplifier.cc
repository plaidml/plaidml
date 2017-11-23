#include "tile/lang/simplifier.h"

#include "tile/lang/emitc.h"
#include "tile/lang/semtree.h"

namespace vertexai {
namespace tile {

namespace sem {

class Simplifier : public Visitor {
 public:
  Simplifier() {}

  void Visit(const IntConst& node) override {}

  void Visit(const FloatConst& node) override {}

  void Visit(const LookupLVal& node) override {}

  void Visit(const LoadExpr& node) override { node.inner->Accept(*this); }

  void Visit(const StoreStmt& node) override {
    node.lhs->Accept(*this);
    const_cast<StoreStmt&>(node).rhs = EvalExpr(node.rhs);
  }

  void Visit(const SubscriptLVal& node) override {
    node.ptr->Accept(*this);
    const_cast<SubscriptLVal&>(node).offset = EvalExpr(node.offset);
  }

  void Visit(const DeclareStmt& node) override {
    if (node.init) {
      const_cast<DeclareStmt&>(node).init = EvalExpr(node.init);
    }
  }

  void Visit(const UnaryExpr& node) override { const_cast<UnaryExpr&>(node).inner = EvalExpr(node.inner); }

  void Visit(const BinaryExpr& node) override {
    const_cast<BinaryExpr&>(node).lhs = EvalExpr(node.lhs);
    const_cast<BinaryExpr&>(node).rhs = EvalExpr(node.rhs);

    if (node.op == "*") {
      if (CheckIntConstValue(node.rhs, 1)) {
        // Check for (L * 1), return (L)
        new_expr_ = node.lhs;
      } else if (CheckIntConstValue(node.lhs, 1)) {
        // Check for (1 * R), return (R)
        new_expr_ = node.rhs;
      } else if (CheckIntConstValue(node.lhs, 0)) {
        // Check for (0 * R), return (0)
        new_expr_ = node.lhs;
      } else if (CheckIntConstValue(node.rhs, 0)) {
        // Check for (L * 0), return (0)
        new_expr_ = node.rhs;
      }
    } else if (node.op == "/") {
      if (CheckIntConstValue(node.rhs, 1)) {
        // Check for (L / 1), return (L)
        new_expr_ = node.lhs;
      } else if (CheckIntConstValue(node.lhs, 0)) {
        // Check for (0 / R), return (0)
        new_expr_ = node.lhs;
      }
    } else if (node.op == "+") {
      if (CheckIntConstValue(node.rhs, 0)) {
        // Check for (L + 0), return (L)
        new_expr_ = node.lhs;
      } else if (CheckIntConstValue(node.lhs, 0)) {
        // Check for (0 + R), return (R)
        new_expr_ = node.rhs;
      }
    } else if (node.op == "-") {
      if (CheckIntConstValue(node.rhs, 0)) {
        // Check for (L - 0), return (L)
        new_expr_ = node.lhs;
      }
    }
  }

  void Visit(const CondExpr& node) override {
    const_cast<CondExpr&>(node).cond = EvalExpr(node.cond);
    const_cast<CondExpr&>(node).tcase = EvalExpr(node.tcase);
    const_cast<CondExpr&>(node).fcase = EvalExpr(node.fcase);
  }

  void Visit(const SelectExpr& node) override {
    const_cast<SelectExpr&>(node).cond = EvalExpr(node.cond);
    const_cast<SelectExpr&>(node).tcase = EvalExpr(node.tcase);
    const_cast<SelectExpr&>(node).fcase = EvalExpr(node.fcase);
  }

  void Visit(const ClampExpr& node) override {
    const_cast<ClampExpr&>(node).val = EvalExpr(node.val);
    const_cast<ClampExpr&>(node).min = EvalExpr(node.min);
    const_cast<ClampExpr&>(node).max = EvalExpr(node.max);
  }

  void Visit(const CastExpr& node) override { const_cast<CastExpr&>(node).val = EvalExpr(node.val); }

  void Visit(const CallExpr& node) override {
    const_cast<CallExpr&>(node).func = EvalExpr(node.func);
    for (size_t i = 0; i < node.vals.size(); i++) {
      const_cast<CallExpr&>(node).vals[i] = EvalExpr(node.vals[i]);
    }
  }

  void Visit(const LimitConst& node) override {}

  void Visit(const IndexExpr& node) override {}

  void Visit(const Block& node) override {
    for (size_t i = 0; i < node.statements.size(); i++) {
      const_cast<Block&>(node).statements[i] = EvalStmt(node.statements[i]);
    }
  }

  void Visit(const IfStmt& node) override {
    const_cast<IfStmt&>(node).cond = EvalExpr(node.cond);
    if (node.iftrue) {
      const_cast<IfStmt&>(node).iftrue = EvalStmt(node.iftrue);
    }
    if (node.iffalse) {
      const_cast<IfStmt&>(node).iffalse = EvalStmt(node.iffalse);
    }
  }

  void Visit(const ForStmt& node) override { const_cast<ForStmt&>(node).inner = EvalStmt(node.inner); }

  void Visit(const WhileStmt& node) override {
    const_cast<WhileStmt&>(node).cond = EvalExpr(node.cond);
    const_cast<WhileStmt&>(node).inner = EvalStmt(node.inner);
  }

  void Visit(const BarrierStmt& node) override {}

  void Visit(const ReturnStmt& node) override {
    if (node.value) {
      const_cast<ReturnStmt&>(node).value = EvalExpr(node.value);
    }
  }

  void Visit(const Function& node) override { const_cast<Function&>(node).body = EvalStmt(node.body); }

 private:
  bool CheckIntConstValue(const ExprPtr& expr, int64_t value) {
    auto int_const = std::dynamic_pointer_cast<IntConst>(expr);
    return (int_const && int_const->value == value);
  }

  ExprPtr EvalExpr(const ExprPtr& expr) {
    Simplifier eval;
    expr->Accept(eval);
    if (eval.new_expr_) {
      return eval.new_expr_;
    }
    return expr;
  }

  StmtPtr EvalStmt(const StmtPtr& stmt) {
    Simplifier eval;
    stmt->Accept(eval);
    if (eval.new_stmt_) {
      return eval.new_stmt_;
    }
    return stmt;
  }

  ExprPtr new_expr_;
  StmtPtr new_stmt_;
};

}  // namespace sem

namespace lang {

void Simplify(const std::vector<KernelInfo>& kernels) {
  for (const auto& ki : kernels) {
    if (VLOG_IS_ON(4)) {
      lang::EmitDebug emit_debug;
      emit_debug.Visit(*ki.kfunc);
      VLOG(4) << "Generic debug kernel before simplification:";
      VLOG(4) << ki.comments;
      VLOG(4) << emit_debug.str();
    }
    sem::Simplifier simplifier;
    ki.kfunc->Accept(simplifier);
  }
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
