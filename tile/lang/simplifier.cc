#include "tile/lang/simplifier.h"

#include "tile/lang/emitc.h"
#include "tile/lang/scope.h"
#include "tile/lang/semtree.h"

namespace vertexai {
namespace tile {
namespace sem {

// This struct holds two totally disparate things, but it's easier
// to use a single scope object throughout the recursive Simplifier
// rather than having to pass multiple scope objects down.
struct Symbol {
  // This member is set if the symbol can be resolved to an IntConst value.
  // If such a value exists, substitute in an IntConst over a LoadExpr.
  boost::optional<int64_t> const_value;

  // This member is set if the symbol can be resolved to another symbol.
  // That is, if the code looks like:
  //   int x = y;
  // Then symbol 'x' is an alias to 'y'.
  // If such an alias exists, substitute every use for 'x' with a LoadExpr of 'y'.
  // It's also safe to elide this declaration since every use of 'x' will be replaced with 'y'.
  boost::optional<std::string> alias;
};

class Simplifier : public boost::static_visitor<> {
 public:
  explicit Simplifier(lang::Scope<Symbol>* scope) : scope_{scope} {}

  void operator()(IntConst& node) {}  // NOLINT

  void operator()(FloatConst& node) {}  // NOLINT

  void operator()(LookupLVal& node) {  // NOLINT
    auto symbol = scope_->Lookup(node.name);
    // Check if a symbol exists that is an alias for another symbol.
    if (symbol && (*symbol).alias) {
      // If such a symbol exists, substitute it in.
      ref_ = *(*symbol).alias;
      const_cast<LookupLVal&>(node).name = ref_;
    } else {
      ref_ = node.name;
    }
  }

  void operator()(LoadExpr& node) {  // NOLINT
    auto ref = Resolve(node.inner);
    auto symbol = scope_->Lookup(ref);
    // Check if a symbol exists that refers to an IntConst value.
    if (symbol && (*symbol).const_value) {
      // If such a symbol exists, substitute the IntConst expr in directly.
      new_expr_ = std::make_shared<Expression>(IntConst(*(*symbol).const_value));
    }
  }

  void operator()(StoreStmt& node) {  // NOLINT
    Resolve(node.lhs);
    node.rhs = EvalExpr(node.rhs);
  }

  void operator()(SubscriptLVal& node) {  // NOLINT
    ref_ = Resolve(node.ptr);
    node.offset = EvalExpr(node.offset);
  }

  void operator()(DeclareStmt& node) {  // NOLINT
    if (node.init) {
      auto init = EvalExpr(node.init);

      auto int_const = boost::get<IntConst>(init.get());
      if (int_const) {
        Symbol symbol;
        symbol.const_value = int_const->value;
        scope_->Bind(node.name, symbol);
        // Mark this statement as elided.
        new_stmt_ = std::make_shared<Statement>(Block());
        return;
      }

      auto load_expr = boost::get<LoadExpr>(init.get());
      if (load_expr) {
        auto lookup = boost::get<LookupLVal>(load_expr->inner.get());
        if (lookup) {
          auto ref = Resolve(load_expr->inner);
          Symbol symbol;
          symbol.alias = ref;
          scope_->Bind(node.name, symbol);
          // Mark this statement as elided.
          new_stmt_ = std::make_shared<Statement>(Block());
          return;
        }
      }

      node.init = init;
    }
  }

  void operator()(UnaryExpr& node) { node.inner = EvalExpr(node.inner); }  // NOLINT

  void operator()(BinaryExpr& node) {  // NOLINT
    node.lhs = EvalExpr(node.lhs);
    node.rhs = EvalExpr(node.rhs);

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

  void operator()(CondExpr& node) {  // NOLINT
    node.cond = EvalExpr(node.cond);
    node.tcase = EvalExpr(node.tcase);
    node.fcase = EvalExpr(node.fcase);
  }

  void operator()(SelectExpr& node) {  // NOLINT
    node.cond = EvalExpr(node.cond);
    node.tcase = EvalExpr(node.tcase);
    node.fcase = EvalExpr(node.fcase);
  }

  void operator()(ClampExpr& node) {  // NOLINT
    node.val = EvalExpr(node.val);
    node.min = EvalExpr(node.min);
    node.max = EvalExpr(node.max);
  }

  void operator()(CastExpr& node) { node.val = EvalExpr(node.val); }  // NOLINT

  void operator()(CallExpr& node) {  // NOLINT
    node.func = EvalExpr(node.func);
    for (size_t i = 0; i < node.vals.size(); i++) {
      node.vals[i] = EvalExpr(node.vals[i]);
    }
  }

  void operator()(LimitConst& node) {}  // NOLINT

  void operator()(IndexExpr& node) {}  // NOLINT

  void operator()(Block& node) {  // NOLINT
    lang::Scope<Symbol> scope{scope_};
    Block new_block;
    for (const auto& stmt : *node.statements) {
      auto new_stmt = EvalStmt(stmt, &scope);
      auto block = boost::get<Block>(new_stmt.get());
      if (!block || !block->statements->empty()) {
        // Only emit statements that haven't been elided.
        new_block.append(new_stmt);
      }
    }
    new_stmt_ = std::make_shared<Statement>(std::move(new_block));
  }

  void operator()(IfStmt& node) {  // NOLINT
    node.cond = EvalExpr(node.cond);
    if (node.iftrue) {
      (*this)(*node.iftrue);
    }
    if (node.iffalse) {
      (*this)(*node.iffalse);
    }
  }

  void operator()(ForStmt& node) { (*this)(*node.inner); }  // NOLINT

  void operator()(WhileStmt& node) {  // NOLINT
    node.cond = EvalExpr(node.cond);
    (*this)(*node.inner);
  }

  void operator()(BarrierStmt& node) {}  // NOLINT

  void operator()(ReturnStmt& node) {  // NOLINT
    if (node.value) {
      node.value = EvalExpr(node.value);
    }
  }

  void operator()(Function& node) { (*this)(*node.body); }  // NOLINT

 private:
  bool CheckIntConstValue(const ExprPtr& expr, int64_t value) {
    auto int_const = boost::get<IntConst>(expr.get());
    return (int_const && int_const->value == value);
  }

  ExprPtr EvalExpr(const ExprPtr& expr) {
    Simplifier eval{scope_};
    boost::apply_visitor(eval, *expr);
    if (eval.new_expr_) {
      return eval.new_expr_;
    }
    return expr;
  }

  StmtPtr EvalStmt(const StmtPtr& stmt) { return EvalStmt(stmt, scope_); }

  StmtPtr EvalStmt(const StmtPtr& stmt, lang::Scope<Symbol>* scope) {
    Simplifier eval{scope};
    boost::apply_visitor(eval, *stmt);
    if (eval.new_stmt_) {
      return eval.new_stmt_;
    }
    return stmt;
  }

  std::string Resolve(const LValPtr& ptr) {
    Simplifier eval(scope_);
    boost::apply_visitor(eval, *ptr);
    return eval.ref_;
  }

 private:
  ExprPtr new_expr_;
  StmtPtr new_stmt_;
  std::string ref_;

  lang::Scope<Symbol>* scope_;
};

}  // namespace sem

namespace lang {

void Simplify(const std::vector<KernelInfo>& kernels) {
  for (const auto& ki : kernels) {
    if (VLOG_IS_ON(4)) {
      VLOG(4) << "Generic debug kernel before simplification:";
      VLOG(4) << ki.comments;
      VLOG(4) << to_string(*ki.kfunc);
    }
    lang::Scope<sem::Symbol> scope;
    sem::Simplifier simplifier{&scope};
    simplifier(*ki.kfunc);
  }
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
