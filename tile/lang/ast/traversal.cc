// Copyright 2019 Intel Corporation.

#include "tile/lang/ast/traversal.h"

#include <memory>
#include <unordered_map>

#include "base/util/stream_container.h"
#include "tile/lang/ast/fold.h"

namespace vertexai {
namespace tile {
namespace lang {
namespace ast {

class AstTraversal : public AstVisitor<void> {
 public:
  explicit AstTraversal(const ProgramMutations& mutations) {
    for (const auto& expr : mutations.outputs) {
      Push(expr);
    }
    for (const auto& update : mutations.updates) {
      Push(update.src);
      Push(update.dst);
    }
    while (stack_.size()) {
      auto entry = stack_.top();
      stack_.pop();
      if (entry.second) {
        flat_.push_back(entry.first);
      } else if (!seen_.count(entry.first.get())) {
        seen_.insert(entry.first.get());
        stack_.push(std::make_pair(entry.first, true));
        entry.first->Accept(this);
      }
    }
    IVLOG(4, "AstTraversal: " << StreamContainer(flat_));
  }

  const std::vector<ExprPtr>& flat() const { return flat_; }

 private:
  void Visit(const CallExpr& expr) final {
    // push arguments from right-to-left so they eventually get processed in left-to-right order
    for (auto it = expr.args.rbegin(); it != expr.args.rend(); ++it) {
      Push(*it);
    }
  }

  void Visit(const ContractionExpr& expr) final {
    // push inputs from right-to-left so they eventually get processed in left-to-right order
    IVLOG(6, "Visiting ContractionExpr: " << &expr);
    IVLOG(6, "  with agg_op " << to_string(expr.agg_op) << ", combo_op " << to_string(expr.combo_op));
    for (auto it = expr.srcs.rbegin(); it != expr.srcs.rend(); ++it) {
      Push((*it)->ref);
    }
    if (expr.use_default) {
      Push(expr.use_default);
    }
  }

  void Visit(const DimExprExpr& expr) final {}
  void Visit(const FloatConst& expr) final {}
  void Visit(const IntConst& expr) final {}
  void Visit(const ParamExpr& expr) final {}
  void Visit(const GradOverrideExpr& expr) final {}

 private:
  void Push(const ExprPtr& expr) {
    if (!expr) {
      throw std::runtime_error("Invalid expression in AstTraversal::Push");
    }
    IVLOG(4, "AstTraversal::Push> " << expr.get());
    stack_.push(std::make_pair(expr, false));
  }

 private:
  std::stack<std::pair<ExprPtr, bool>> stack_;
  std::vector<ExprPtr> flat_;
  std::unordered_set<const Expr*> seen_;
};

std::vector<ExprPtr> FlattenAst(const ProgramMutations& mutations) {
  AstTraversal traversal(mutations);
  return traversal.flat();
}

class AstPassRunner : AstVisitor<void> {
 public:
  AstPassRunner(                          //
      const std::vector<ExprPtr>& ast,    //
      const ProgramMutations& mutations,  //
      AstPass* pass)
      : pass_(pass) {
    for (const auto& expr : ast) {
      expr->Accept(this);
    }
    for (const auto& output : mutations.outputs) {
      auto new_output = Translate(output);
      IVLOG(4, "AstPassRunner> output: " << output << " -> " << new_output);
      mutations_.outputs.emplace_back(new_output);
    }
    for (const auto& update : mutations.updates) {
      ProgramUpdate new_update{Translate(update.src), Translate(update.dst)};
      IVLOG(4, "AstPassRunner> update.src: " << update.src << " -> " << new_update.src);
      IVLOG(4, "AstPassRunner> update.dst: " << update.dst << " -> " << new_update.dst);
      mutations_.updates.emplace_back(new_update);
    }
  }

  const ProgramMutations& mutations() const { return mutations_; }

 private:
  void Visit(const CallExpr& expr) final {
    IVLOG(4, "AstPassRunner::Visit(CallExpr)> " << &expr);
    auto new_expr = std::make_shared<CallExpr>(expr.fn, expr.args);
    for (size_t i = 0; i < expr.args.size(); i++) {
      new_expr->args[i] = Translate(expr.args[i]);
    }
    new_expr->ComputeShape();
    GenericVisit(expr, *new_expr);
  }

  void Visit(const ContractionExpr& expr) final {
    IVLOG(4, "AstPassRunner::Visit(ContractionExpr)> " << &expr);
    auto new_expr = std::make_shared<ContractionExpr>();
    *new_expr = expr;
    for (size_t i = 0; i < expr.srcs.size(); i++) {
      new_expr->srcs[i]->ref = Translate(expr.srcs[i]->ref);
    }
    if (expr.use_default) {
      new_expr->use_default = Translate(expr.use_default);
    }
    new_expr->ComputeShape(expr.shape.layout);
    GenericVisit(expr, *new_expr);
  }

  void Visit(const DimExprExpr& expr) final {
    IVLOG(4, "AstPassRunner::Visit(DimExprExpr)> " << &expr);
    GenericVisit(expr, expr);
  }

  void Visit(const FloatConst& expr) final {
    IVLOG(4, "AstPassRunner::Visit(FloatConst)> " << &expr);
    GenericVisit(expr, expr);
  }

  void Visit(const IntConst& expr) final {
    IVLOG(4, "AstPassRunner::Visit(IntConst)> " << &expr);
    GenericVisit(expr, expr);
  }

  void Visit(const ParamExpr& expr) final {
    IVLOG(4, "AstPassRunner::Visit(ParamExpr)> " << &expr);
    GenericVisit(expr, expr);
  }

  // TODO
  void Visit(const GradOverrideExpr& expr) final {
    IVLOG(4, "AstPassRunner::Visit(GradOverrideExpr)> " << &expr);
    GenericVisit(expr, expr);
  }

 private:
  template <typename T>
  void GenericVisit(const T& old_expr, const T& expr) {
    auto new_expr = pass_->Visit(expr);
    if (new_expr.get() != &old_expr) {
      rewrites_.emplace(&old_expr, new_expr);
    }
  }

  ExprPtr Translate(const ExprPtr& expr) {
    auto it = rewrites_.find(expr.get());
    if (it == rewrites_.end()) {
      return expr;
    }
    return it->second;
  }

 private:
  AstPass* pass_;
  std::unordered_map<const Expr*, ExprPtr> rewrites_;
  ProgramMutations mutations_;
};

ProgramMutations RunAstPass(const ProgramMutations& mutations, AstPass* pass) {
  auto ast = FlattenAst(mutations);
  AstPassRunner runner(ast, mutations, pass);
  return runner.mutations();
}

}  // namespace ast
}  // namespace lang
}  // namespace tile
}  // namespace vertexai
