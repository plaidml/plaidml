
#include "tile/lang/sym_poly.h"

#include <memory>

#include "tile/lang/compose.h"
#include "tile/lang/type.h"

namespace vertexai {
namespace tile {
namespace lang {

class LiteralPolynomial : public SymbolicPolynomial {
 public:
  explicit LiteralPolynomial(int64_t value) : value_(value) {}
  SymbolicPolynomialPtr Xify() const override { return std::make_shared<LiteralPolynomial>(value_); }
  SymbolicPolynomialPtr DeXify() const override { return std::make_shared<LiteralPolynomial>(value_); }
  SymbolicPolynomialPtr Compose(const FunctionApplication& fa) const override {
    return std::make_shared<LiteralPolynomial>(value_);
  }
  SymbolicPolynomialPtr Decompose(BoundFunction* bf) const override {
    return std::make_shared<LiteralPolynomial>(value_);
  }
  Polynomial Evaluate(const Bindings& bindings) const override { return Polynomial(value_); }
  std::string ToString() const override { return std::to_string(value_); }

 private:
  int64_t value_;
};

class LookupPolynomial : public SymbolicPolynomial {
 public:
  explicit LookupPolynomial(const std::string& name) : name_(name) {}
  SymbolicPolynomialPtr Xify() const override { return std::make_shared<LookupPolynomial>("X" + name_); }
  SymbolicPolynomialPtr DeXify() const override {
    if (name_.size() < 1 || name_[0] != 'X') {
      throw std::runtime_error("Failure to DeXify in LookupPolynomial");
    }
    return std::make_shared<LookupPolynomial>(name_.substr(1, name_.size() - 1));
  }
  SymbolicPolynomialPtr Compose(const FunctionApplication& fa) const override;
  SymbolicPolynomialPtr Decompose(BoundFunction* bf) const override {
    throw std::runtime_error("Decompose not implemented for LookupPolynomial, lookup value = " + name_);
  }
  Polynomial Evaluate(const Bindings& bindings) const override {
    auto it = bindings.find(name_);
    if (it == bindings.end()) {
      throw std::runtime_error("Unknown variable " + name_ + " in polynomial");
    }
    if (it->second.tag != Binding::ICONST) {
      throw std::runtime_error("Variable " + name_ +
                               " used in a polynomial which requires it to be a constant integer");
    }
    return Polynomial(it->second.iconst);
  }
  std::string ToString() const override { return name_; }

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
    return std::make_shared<LookupPolynomial>(bf->Apply(value_));
  }
  Polynomial Evaluate(const Bindings& bindings) const override {
    throw std::runtime_error("Evaluate not implemented for ValuePolynomial");
  }
  std::string ToString() const override { throw std::runtime_error("ToString not implemented for ValuePolynomial"); }

 private:
  std::shared_ptr<Value> value_;
};

SymbolicPolynomialPtr LookupPolynomial::Compose(const FunctionApplication& fa) const {
  auto it = fa.bindings_.find(name_);
  if (it == fa.bindings_.end()) {
    throw std::runtime_error("Unknown variable " + name_ + " in polynomial");
  }
  return std::make_shared<ValuePolynomial>(it->second);
}

class IndexPolynomial : public SymbolicPolynomial {
 public:
  explicit IndexPolynomial(const std::string& index) : index_(index) {}
  SymbolicPolynomialPtr Xify() const override { return std::make_shared<IndexPolynomial>(index_); }
  SymbolicPolynomialPtr DeXify() const override { return std::make_shared<IndexPolynomial>(index_); }
  SymbolicPolynomialPtr Compose(const FunctionApplication& fa) const override {
    return std::make_shared<IndexPolynomial>(index_);
  }
  SymbolicPolynomialPtr Decompose(BoundFunction* bf) const override {
    return std::make_shared<IndexPolynomial>(index_);
  }
  Polynomial Evaluate(const Bindings& bindings) const override { return Polynomial(index_); }
  std::string ToString() const override { return index_; }

 private:
  std::string index_;
};

class UnaryOpPolynomial : public SymbolicPolynomial {
 public:
  UnaryOpPolynomial(const std::string& op, const SymbolicPolynomialPtr& val) : op_(op), val_(val) {}
  SymbolicPolynomialPtr Xify() const override { return std::make_shared<UnaryOpPolynomial>(op_, val_->Xify()); }
  SymbolicPolynomialPtr DeXify() const override { return std::make_shared<UnaryOpPolynomial>(op_, val_->DeXify()); }
  SymbolicPolynomialPtr Compose(const FunctionApplication& fa) const override {
    return std::make_shared<UnaryOpPolynomial>(op_, val_->Compose(fa));
  }
  SymbolicPolynomialPtr Decompose(BoundFunction* bf) const override {
    return std::make_shared<UnaryOpPolynomial>(op_, val_->Decompose(bf));
  }
  Polynomial Evaluate(const Bindings& bindings) const override {
    if (op_ != "-") {
      throw std::runtime_error("Unknown unary polynomial op");
    }
    return -val_->Evaluate(bindings);
  }
  std::string ToString() const override { return "(-" + val_->ToString() + ")"; }

 private:
  std::string op_;
  SymbolicPolynomialPtr val_;
};

class BinaryOpPolynomial : public SymbolicPolynomial {
 public:
  BinaryOpPolynomial(const std::string& op, const SymbolicPolynomialPtr& lhs, const SymbolicPolynomialPtr& rhs)
      : op_(op), lhs_(lhs), rhs_(rhs) {}
  SymbolicPolynomialPtr Xify() const override {
    return std::make_shared<BinaryOpPolynomial>(op_, lhs_->Xify(), rhs_->Xify());
  }
  SymbolicPolynomialPtr DeXify() const override {
    return std::make_shared<BinaryOpPolynomial>(op_, lhs_->DeXify(), rhs_->DeXify());
  }
  SymbolicPolynomialPtr Compose(const FunctionApplication& fa) const override {
    return std::make_shared<BinaryOpPolynomial>(op_, lhs_->Compose(fa), rhs_->Compose(fa));
  }
  SymbolicPolynomialPtr Decompose(BoundFunction* bf) const override {
    return std::make_shared<BinaryOpPolynomial>(op_, lhs_->Decompose(bf), rhs_->Decompose(bf));
  }
  Polynomial Evaluate(const Bindings& bindings) const override {
    if (op_ == "+") {
      return lhs_->Evaluate(bindings) + rhs_->Evaluate(bindings);
    }
    if (op_ == "-") {
      return lhs_->Evaluate(bindings) - rhs_->Evaluate(bindings);
    }
    if (op_ == "*") {
      Polynomial lhs = lhs_->Evaluate(bindings);
      Polynomial rhs = rhs_->Evaluate(bindings);
      if (lhs.isConstant()) {
        return rhs * lhs.constant();
      }
      if (rhs.isConstant()) {
        return lhs * rhs.constant();
      }
      throw std::runtime_error("Non-linear polynomial");
    }
    if (op_ == "/") {
      Polynomial lhs = lhs_->Evaluate(bindings);
      Polynomial rhs = rhs_->Evaluate(bindings);
      if (!rhs.isConstant()) {
        throw std::runtime_error("Divisor of polynomials must be a constant");
      }
      return lhs / rhs.constant();
    }
    throw std::runtime_error("Unknown binary polynomial op");
  }
  std::string ToString() const override { return "(" + lhs_->ToString() + " " + op_ + " " + rhs_->ToString() + ")"; }

 private:
  std::string op_;
  SymbolicPolynomialPtr lhs_;
  SymbolicPolynomialPtr rhs_;
};

SymbolicPolynomialPtr SymbolicPolynomial::MakeLiteral(int64_t value) {
  return std::make_shared<LiteralPolynomial>(value);
}

SymbolicPolynomialPtr SymbolicPolynomial::MakeSymbol(const std::string& sym) {
  return std::make_shared<LookupPolynomial>(sym);
}

SymbolicPolynomialPtr SymbolicPolynomial::MakeIndex(const std::string& idx) {
  return std::make_shared<IndexPolynomial>(idx);
}

SymbolicPolynomialPtr SymbolicPolynomial::MakeUnaryOp(const std::string& op, const SymbolicPolynomialPtr& val) {
  return std::make_shared<UnaryOpPolynomial>(op, val);
}

SymbolicPolynomialPtr SymbolicPolynomial::MakeBinaryOp(const std::string& op, const SymbolicPolynomialPtr& lhs,
                                                       const SymbolicPolynomialPtr& rhs) {
  return std::make_shared<BinaryOpPolynomial>(op, lhs, rhs);
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
