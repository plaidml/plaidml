
#include "tile/lang/sym_poly.h"

#include <memory>

#include "base/util/intern.h"
#include "tile/lang/compose.h"
#include "tile/lang/type.h"

namespace vertexai {
namespace tile {
namespace lang {

using namespace math;  // NOLINT

class LiteralPolynomial : public SymbolicPolynomial {
 public:
  explicit LiteralPolynomial(int64_t value) : value_(value) {}
  SymbolicPolynomialPtr Xify() const override { return MakeLiteral(value_); }
  SymbolicPolynomialPtr DeXify() const override { return MakeLiteral(value_); }
  SymbolicPolynomialPtr Compose(const FunctionApplication& fa) const override { return MakeLiteral(value_); }
  SymbolicPolynomialPtr Decompose(BoundFunction* bf) const override { return MakeLiteral(value_); }
  Polynomial<Rational> Evaluate(const Bindings& bindings) const override { return Polynomial<Rational>(value_); }
  std::string ToString() const override { return std::to_string(value_); }
  std::shared_ptr<Value> value() const override { return std::shared_ptr<Value>{}; }
  SymbolicSpec subspec() const override { return SymbolicSpec{}; }

 private:
  int64_t value_;
};

class LookupPolynomial : public SymbolicPolynomial {
 public:
  explicit LookupPolynomial(const std::string& name) : name_(name) {}
  SymbolicPolynomialPtr Xify() const override { return Interned<LookupPolynomial>::make("X" + name_); }
  SymbolicPolynomialPtr DeXify() const override {
    if (name_.size() < 1 || name_[0] != 'X') {
      throw std::runtime_error("Failure to DeXify in LookupPolynomial");
    }
    return Interned<LookupPolynomial>::make(name_.substr(1, name_.size() - 1));
  }
  SymbolicPolynomialPtr Compose(const FunctionApplication& fa) const override;
  SymbolicPolynomialPtr Decompose(BoundFunction* bf) const override {
    throw std::runtime_error("Decompose not implemented for LookupPolynomial, lookup value = " + name_);
  }
  Polynomial<Rational> Evaluate(const Bindings& bindings) const override {
    auto it = bindings.find(name_);
    if (it == bindings.end()) {
      throw std::runtime_error("Unknown variable " + name_ + " in polynomial");
    }
    if (it->second.tag != Binding::ICONST) {
      throw std::runtime_error("Variable " + name_ +
                               " used in a polynomial which requires it to be a constant integer");
    }
    return Polynomial<Rational>(it->second.iconst);
  }
  std::string ToString() const override { return name_; }
  std::shared_ptr<Value> value() const override { return std::shared_ptr<Value>{}; }
  SymbolicSpec subspec() const override { return SymbolicSpec{}; }

 private:
  std::string name_;
};

class ValuePolynomial : public SymbolicPolynomial {
 public:
  explicit ValuePolynomial(const std::shared_ptr<Value>& value) : value_(value) {}
  SymbolicPolynomialPtr Xify() const override { throw std::runtime_error("Xify not implemented for ValuePolynomial"); }
  SymbolicPolynomialPtr DeXify() const override {
    throw std::runtime_error("DeXify not implemented for ValuePolynomial");
  }
  SymbolicPolynomialPtr Compose(const FunctionApplication& fa) const override {
    throw std::runtime_error("Compose not implemented for ValuePolynomial");
  }
  SymbolicPolynomialPtr Decompose(BoundFunction* bf) const override {
    return Interned<LookupPolynomial>::make(bf->Apply(value_));
  }
  Polynomial<Rational> Evaluate(const Bindings& bindings) const override {
    throw std::runtime_error("Evaluate not implemented for ValuePolynomial");
  }
  std::string ToString() const override { throw std::runtime_error("ToString not implemented for ValuePolynomial"); }
  std::shared_ptr<Value> value() const override { return std::shared_ptr<Value>{value_}; }
  SymbolicSpec subspec() const override { return SymbolicSpec{}; }

 private:
  std::shared_ptr<Value> value_;
};

SymbolicPolynomialPtr LookupPolynomial::Compose(const FunctionApplication& fa) const {
  auto it = fa.bindings_.find(name_);
  if (it == fa.bindings_.end()) {
    throw std::runtime_error("Unknown variable " + name_ + " in polynomial");
  }
  return Interned<ValuePolynomial>::make(it->second);
}

class IndexPolynomial : public SymbolicPolynomial {
 public:
  explicit IndexPolynomial(const std::string& index) : index_(index) {}
  SymbolicPolynomialPtr Xify() const override { return MakeIndex(index_); }
  SymbolicPolynomialPtr DeXify() const override { return MakeIndex(index_); }
  SymbolicPolynomialPtr Compose(const FunctionApplication& fa) const override { return MakeIndex(index_); }
  SymbolicPolynomialPtr Decompose(BoundFunction* bf) const override { return MakeIndex(index_); }
  Polynomial<Rational> Evaluate(const Bindings& bindings) const override { return Polynomial<Rational>(index_); }
  std::string ToString() const override { return index_; }
  std::shared_ptr<Value> value() const override { return std::shared_ptr<Value>{}; }
  SymbolicSpec subspec() const override { return SymbolicSpec{}; }

 private:
  std::string index_;
};

class UnaryOpPolynomial : public SymbolicPolynomial {
 public:
  UnaryOpPolynomial(const std::string& op, const SymbolicPolynomialPtr& val) : op_(op), val_(val) {}
  SymbolicPolynomialPtr Xify() const override { return MakeUnaryOp(op_, val_->Xify()); }
  SymbolicPolynomialPtr DeXify() const override { return MakeUnaryOp(op_, val_->DeXify()); }
  SymbolicPolynomialPtr Compose(const FunctionApplication& fa) const override {
    return MakeUnaryOp(op_, val_->Compose(fa));
  }
  SymbolicPolynomialPtr Decompose(BoundFunction* bf) const override { return MakeUnaryOp(op_, val_->Decompose(bf)); }
  Polynomial<Rational> Evaluate(const Bindings& bindings) const override {
    if (op_ != "-") {
      throw std::runtime_error("Unknown unary polynomial op");
    }
    return -val_->Evaluate(bindings);
  }
  std::string ToString() const override { return "(-" + val_->ToString() + ")"; }
  std::shared_ptr<Value> value() const override { return std::shared_ptr<Value>{}; }
  SymbolicSpec subspec() const override { return SymbolicSpec{val_}; }

 private:
  std::string op_;
  SymbolicPolynomialPtr val_;
};

class BinaryOpPolynomial : public SymbolicPolynomial {
 public:
  BinaryOpPolynomial(const std::string& op, const SymbolicPolynomialPtr& lhs, const SymbolicPolynomialPtr& rhs)
      : op_(op), lhs_(lhs), rhs_(rhs) {}
  SymbolicPolynomialPtr Xify() const override { return MakeBinaryOp(op_, lhs_->Xify(), rhs_->Xify()); }
  SymbolicPolynomialPtr DeXify() const override { return MakeBinaryOp(op_, lhs_->DeXify(), rhs_->DeXify()); }
  SymbolicPolynomialPtr Compose(const FunctionApplication& fa) const override {
    return MakeBinaryOp(op_, lhs_->Compose(fa), rhs_->Compose(fa));
  }
  SymbolicPolynomialPtr Decompose(BoundFunction* bf) const override {
    return MakeBinaryOp(op_, lhs_->Decompose(bf), rhs_->Decompose(bf));
  }
  Polynomial<Rational> Evaluate(const Bindings& bindings) const override {
    if (op_ == "+") {
      return lhs_->Evaluate(bindings) + rhs_->Evaluate(bindings);
    }
    if (op_ == "-") {
      return lhs_->Evaluate(bindings) - rhs_->Evaluate(bindings);
    }
    if (op_ == "*") {
      Polynomial<Rational> lhs = lhs_->Evaluate(bindings);
      Polynomial<Rational> rhs = rhs_->Evaluate(bindings);
      if (lhs.isConstant()) {
        return rhs * lhs.constant();
      }
      if (rhs.isConstant()) {
        return lhs * rhs.constant();
      }
      throw std::runtime_error("Non-linear polynomial");
    }
    if (op_ == "/") {
      Polynomial<Rational> lhs = lhs_->Evaluate(bindings);
      Polynomial<Rational> rhs = rhs_->Evaluate(bindings);
      if (!rhs.isConstant()) {
        throw std::runtime_error("Divisor of polynomials must be a constant");
      }
      return lhs / rhs.constant();
    }
    throw std::runtime_error("Unknown binary polynomial op");
  }
  std::string ToString() const override { return "(" + lhs_->ToString() + " " + op_ + " " + rhs_->ToString() + ")"; }
  std::shared_ptr<Value> value() const override { return std::shared_ptr<Value>{}; }
  SymbolicSpec subspec() const override { return SymbolicSpec{lhs_, rhs_}; }

 private:
  std::string op_;
  SymbolicPolynomialPtr lhs_;
  SymbolicPolynomialPtr rhs_;
};

SymbolicPolynomialPtr SymbolicPolynomial::MakeLiteral(int64_t value) {
  return Interned<LiteralPolynomial>::make(value);
}

SymbolicPolynomialPtr SymbolicPolynomial::MakeSymbol(const std::string& sym) {
  return Interned<LookupPolynomial>::make(sym);
}

SymbolicPolynomialPtr SymbolicPolynomial::MakeIndex(const std::string& idx) {
  return Interned<IndexPolynomial>::make(idx);
}

SymbolicPolynomialPtr SymbolicPolynomial::MakeUnaryOp(const std::string& op, const SymbolicPolynomialPtr& val) {
  return Interned<UnaryOpPolynomial>::make(op, val);
}

SymbolicPolynomialPtr SymbolicPolynomial::MakeBinaryOp(const std::string& op, const SymbolicPolynomialPtr& lhs,
                                                       const SymbolicPolynomialPtr& rhs) {
  return Interned<BinaryOpPolynomial>::make(op, lhs, rhs);
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
