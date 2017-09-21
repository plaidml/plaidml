

#pragma once

// Semantic tree representing Tile operations: an intermediate representation
// provided to CG backends (LLVM, OpenCL, etc.)

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "tile/lang/shape.h"

namespace vertexai {
namespace tile {
namespace sem {

// Simple typing.  Every type is either a TILE value type, a TILE pointer type, or an index
// TILE value types consist of the underlying type and a vector width
struct Type {
  enum BaseType { TVOID, INDEX, VALUE, POINTER_MUT, POINTER_CONST };
  enum MemoryRegion { NORMAL, LOCAL, GLOBAL };

  Type(BaseType base_in = TVOID, lang::DataType dtype_in = lang::DataType::INVALID, uint64_t vec_width_in = 1,
       uint64_t array_in = 0, MemoryRegion region_in = NORMAL)
      : base{base_in}, dtype{dtype_in}, vec_width{vec_width_in}, array{array_in}, region{region_in} {}

  BaseType base;
  lang::DataType dtype;
  uint64_t vec_width;  // 1 is non-vector
  uint64_t array;      // 0 is non-array, otherwise, array size
  MemoryRegion region;
};

class Visitor;

// All semtree elements are nodes.
struct Node {
  virtual void Accept(Visitor &) const = 0;
};

// Statements have side effects and return void.
struct Statement : public Node {
  virtual bool isBlock() const { return false; }
};
typedef std::shared_ptr<Statement> StmtPtr;

// Expressions have no side effects and return a value.
struct Expression : public Node {};
typedef std::shared_ptr<Expression> ExprPtr;

// LValues can be loaded from or stored into, and have no side effects.
struct LValue : public Node {};
typedef std::shared_ptr<LValue> LValPtr;

// A constant value.
struct IntConst : public Expression {
  int64_t value;
  explicit IntConst(int64_t val) : value(val) {}
  void Accept(Visitor &) const final;
};

struct FloatConst : public Expression {
  double value;
  explicit FloatConst(double val) : value(val) {}
  void Accept(Visitor &) const final;
};

// A symbol table lookup (ie, a variable name)
struct LookupLVal : public LValue {
  std::string name;
  explicit LookupLVal(const std::string &n) : name(n) {}
  void Accept(Visitor &) const final;
};

// A load from an LVAL (variable or pointer)
struct LoadExpr : public Expression {
  LValPtr inner;
  explicit LoadExpr(const LValPtr in) : inner(in) {}
  void Accept(Visitor &) const final;
};

// A store to an LVAL
struct StoreStmt : public Statement {
  LValPtr lhs;
  ExprPtr rhs;
  StoreStmt(const LValPtr l, const ExprPtr r) : lhs(l), rhs(r) {}
  void Accept(Visitor &) const final;
};

// Create LVAL reference to an array element
struct SubscriptLVal : public LValue {
  LValPtr ptr;
  ExprPtr offset;
  SubscriptLVal(const LValPtr p, const ExprPtr o) : ptr(p), offset(o) {}
  void Accept(Visitor &) const final;
};

// A declaration of a new variable (local to scope)
struct DeclareStmt : public Statement {
  Type type;
  std::string name;
  ExprPtr init;
  DeclareStmt(const Type &t, const std::string n, ExprPtr i) : type(t), name(n), init(i) {}
  void Accept(Visitor &) const final;
};

// A unary operator
struct UnaryExpr : public Expression {
  std::string op;
  ExprPtr inner;
  UnaryExpr(std::string o, ExprPtr i) : op(o), inner(i) {}
  void Accept(Visitor &) const final;
};

// A binary operator
struct BinaryExpr : public Expression {
  std::string op;
  ExprPtr lhs;
  ExprPtr rhs;
  BinaryExpr(std::string o, ExprPtr l, ExprPtr r) : op(o), lhs(l), rhs(r) {}
  void Accept(Visitor &) const final;
};

// Conditional operator: evaluates either tcase or fcase
struct CondExpr : public Expression {
  ExprPtr cond;
  ExprPtr tcase;
  ExprPtr fcase;
  CondExpr(ExprPtr c, ExprPtr t, ExprPtr f) : cond(c), tcase(t), fcase(f) {}
  void Accept(Visitor &) const final;
};

// Select operator: evaluates both tcase and fcase
struct SelectExpr : public Expression {
  ExprPtr cond;
  ExprPtr tcase;
  ExprPtr fcase;
  SelectExpr(ExprPtr c, ExprPtr t, ExprPtr f) : cond(c), tcase(t), fcase(f) {}
  void Accept(Visitor &) const final;
};

// Clamp operator: constrains a value within limits
struct ClampExpr : public Expression {
  ExprPtr val;
  ExprPtr min;
  ExprPtr max;
  ClampExpr(ExprPtr v, ExprPtr n, ExprPtr x) : val(v), min(n), max(x) {}
  void Accept(Visitor &) const final;
};

// Type conversion equivalent to static_cast
struct CastExpr : public Expression {
  Type type;
  ExprPtr val;
  CastExpr(const Type &t, ExprPtr v) : type(t), val(v) {}
  void Accept(Visitor &) const final;
};

// A call of a function
struct CallExpr : public Expression {
  ExprPtr func;
  std::vector<ExprPtr> vals;
  CallExpr(ExprPtr f, const std::vector<ExprPtr> &v) : func(f), vals(v) {}
  void Accept(Visitor &) const final;
};

// Represents a type specific constant (min, max, etc)
struct LimitConst : public Expression {
  enum Which { MIN, MAX, ZERO, ONE };
  Which which;
  lang::DataType type;
  LimitConst(Which _which, const lang::DataType &_type) : which(_which), type(_type) {}
  void Accept(Visitor &) const final;
};

// Represents an thread/grid id value
struct IndexExpr : public Expression {
  enum Type { GLOBAL, GROUP, LOCAL };
  Type type;
  size_t dim;
  IndexExpr(Type _type, size_t _dim) : type(_type), dim(_dim) {}
  void Accept(Visitor &) const final;
};

// A block of statements, also a scope for locals
struct Block : public Statement {
  std::vector<StmtPtr> statements;
  Block() {}
  explicit Block(const std::vector<StmtPtr> &s) : statements(s) {}
  bool isBlock() const final { return true; }
  void push_back(StmtPtr p) { statements.push_back(p); }
  void merge(std::shared_ptr<Block> other);
  void append(StmtPtr p);
  void Accept(Visitor &) const final;
};

// An if clause
struct IfStmt : public Statement {
  ExprPtr cond;
  StmtPtr iftrue;
  StmtPtr iffalse;
  IfStmt(ExprPtr c, StmtPtr t, StmtPtr f);
  void Accept(Visitor &) const final;
};

// A highly simplified For statement to allow easier analysis
// Basically, things in the form of:
// for(ssize_t i = 0; i < n*s; i += s) ...
struct ForStmt : public Statement {
  std::string var;
  uint64_t num;
  uint64_t step;
  StmtPtr inner;
  ForStmt(const std::string v, uint64_t n, uint64_t s, StmtPtr i);
  void Accept(Visitor &) const final;
};

// A while loop
struct WhileStmt : public Statement {
  ExprPtr cond;
  StmtPtr inner;
  WhileStmt(ExprPtr c, StmtPtr i);
  void Accept(Visitor &) const final;
};

// A break statement
struct BreakStmt : public Statement {
  BreakStmt() {}
  void Accept(Visitor &) const final;
};

// A continue statement
struct ContinueStmt : public Statement {
  ContinueStmt() {}
  void Accept(Visitor &) const final;
};

// A statement representing an inter-thread barrier
struct BarrierStmt : public Statement {
  BarrierStmt() {}
  void Accept(Visitor &) const final;
};

// A return statement
struct ReturnStmt : public Statement {
  ExprPtr value;
  explicit ReturnStmt(ExprPtr v) : value(v) {}
  void Accept(Visitor &) const final;
};

// A function, note: this isn't a statement or an expression
// This is also the entry point into code generation.
struct Function : public Node {
  typedef std::pair<Type, std::string> param_t;
  typedef std::vector<param_t> params_t;
  std::string name;
  Type ret;
  params_t params;
  StmtPtr body;
  Function() {}
  Function(const std::string n, const Type &r, const params_t &p, StmtPtr b);
  void Accept(Visitor &) const final;
};

class Visitor {
 public:
  virtual void Visit(const IntConst &) = 0;
  virtual void Visit(const FloatConst &) = 0;
  virtual void Visit(const LookupLVal &) = 0;
  virtual void Visit(const LoadExpr &) = 0;
  virtual void Visit(const StoreStmt &) = 0;
  virtual void Visit(const SubscriptLVal &) = 0;
  virtual void Visit(const DeclareStmt &) = 0;
  virtual void Visit(const UnaryExpr &) = 0;
  virtual void Visit(const BinaryExpr &) = 0;
  virtual void Visit(const CondExpr &) = 0;
  virtual void Visit(const SelectExpr &) = 0;
  virtual void Visit(const ClampExpr &) = 0;
  virtual void Visit(const CastExpr &) = 0;
  virtual void Visit(const CallExpr &) = 0;
  virtual void Visit(const LimitConst &) = 0;
  virtual void Visit(const IndexExpr &) = 0;
  virtual void Visit(const Block &) = 0;
  virtual void Visit(const IfStmt &) = 0;
  virtual void Visit(const ForStmt &) = 0;
  virtual void Visit(const WhileStmt &) = 0;
  virtual void Visit(const BreakStmt &) = 0;
  virtual void Visit(const ContinueStmt &) = 0;
  virtual void Visit(const BarrierStmt &) = 0;
  virtual void Visit(const ReturnStmt &) = 0;
  virtual void Visit(const Function &) = 0;
};

}  // namespace sem
}  // namespace tile
}  // namespace vertexai
