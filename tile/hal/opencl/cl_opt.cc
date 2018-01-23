#include "tile/hal/opencl/cl_opt.h"

#include "tile/hal/opencl/exprtype.h"
#include "tile/lang/sembuilder.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

namespace {

// This is an instruction optimizer. It's intended to replace expression patterns with potentially
// more efficient OpenCL builtin functions.
class InsnOptimizer : public sem::Visitor {
 public:
  explicit InsnOptimizer(bool cl_khr_fp16, const hal::proto::HardwareSettings& settings)
      : cl_khr_fp16_{cl_khr_fp16}, settings_{settings} {}

  void Visit(const sem::IntConst& node) override {}

  void Visit(const sem::FloatConst& node) override {}

  void Visit(const sem::LookupLVal& node) override {}

  void Visit(const sem::LoadExpr& node) override {}

  void Visit(const sem::StoreStmt& node) override {}

  void Visit(const sem::SubscriptLVal& node) override {}

  void Visit(const sem::DeclareStmt& node) override {
    using namespace sem::builder;  // NOLINT
    if (node.init) {
      auto add = FindBinaryExpr("+", node.init);
      if (add) {
        auto add_ty = TypeOf(add);
        auto mul = FindBinaryExpr("*", add->rhs);
        if (mul) {
          auto mul_ty = TypeOf(mul);
          // Replace mul/add with a mad() call unless hardware settings have disabled it.
          // Some platforms (i.e. Intel HD Graphics 4000) perform worse with this optimization.
          if (is_float(add_ty.dtype) && is_float(mul_ty.dtype) && !settings_.disable_mad()) {
            auto mad = _("mad")(mul->lhs, mul->rhs, add->lhs);
            const_cast<sem::DeclareStmt&>(node).init = mad;
          }
        }
      }
    }
    scope_->Bind(node.name, node.type);
  }

  void Visit(const sem::UnaryExpr& node) override {}

  void Visit(const sem::BinaryExpr& node) override {}

  void Visit(const sem::CondExpr& node) override {}

  void Visit(const sem::SelectExpr& node) override {}

  void Visit(const sem::ClampExpr& node) override {}

  void Visit(const sem::CastExpr& node) override {}

  void Visit(const sem::CallExpr& node) override {}

  void Visit(const sem::LimitConst& node) override {}

  void Visit(const sem::IndexExpr& node) override {}

  void Visit(const sem::Block& node) override {
    WithScope([&]() {
      for (const auto& stmt : node.statements) {
        EvalStmt(stmt);
      }
    });
  }

  void Visit(const sem::IfStmt& node) override {
    if (node.iftrue) {
      EvalStmt(node.iftrue);
    }
    if (node.iffalse) {
      EvalStmt(node.iffalse);
    }
  }

  void Visit(const sem::ForStmt& node) override {
    WithScope([&]() {
      scope_->Bind(node.var, sem::Type{sem::Type::INDEX});
      EvalStmt(node.inner);
    });
  }

  void Visit(const sem::WhileStmt& node) override {
    WithScope([&]() { EvalStmt(node.inner); });
  }

  void Visit(const sem::BarrierStmt& node) override {}

  void Visit(const sem::ReturnStmt& node) override {}

  void Visit(const sem::Function& node) override {
    lang::Scope<sem::Type> scope;
    scope_ = &scope;
    for (const auto& p : node.params) {
      auto ty = p.first;
      if (ty.dtype == lang::DataType::BOOLEAN) {
        // Global booleans are stored as INT8.
        ty.dtype = lang::DataType::INT8;
      }
      scope.Bind(p.second, ty);
    }
    EvalStmt(node.body);
  }

 private:
  void WithScope(std::function<void()> fn) {
    auto previous_scope = scope_;
    lang::Scope<sem::Type> scope{scope_};
    scope_ = &scope;
    fn();
    scope_ = previous_scope;
  }

  sem::Type TypeOf(const sem::ExprPtr& expr) { return ExprType::TypeOf(scope_, cl_khr_fp16_, expr); }

  void EvalStmt(const sem::StmtPtr& stmt) { stmt->Accept(*this); }

  std::shared_ptr<sem::BinaryExpr> FindBinaryExpr(std::string op, const sem::ExprPtr& expr) {
    auto cast_expr = std::dynamic_pointer_cast<sem::CastExpr>(expr);
    if (cast_expr) {
      return FindBinaryExpr(op, cast_expr->val);
    }
    auto binary_expr = std::dynamic_pointer_cast<sem::BinaryExpr>(expr);
    if (binary_expr && binary_expr->op == op) {
      return binary_expr;
    }
    return nullptr;
  }

 private:
  bool cl_khr_fp16_;
  hal::proto::HardwareSettings settings_;
  lang::Scope<sem::Type>* scope_;
};

}  // namespace

void OptimizeKernel(const lang::KernelInfo& ki, bool cl_khr_fp16, const hal::proto::HardwareSettings& settings) {
  InsnOptimizer opt(cl_khr_fp16, settings);
  ki.kfunc->Accept(opt);
}

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
