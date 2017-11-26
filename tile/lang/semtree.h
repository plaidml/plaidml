// Copyright 2017, Vertex.AI.

#pragma once

// Semantic tree representing Tile operations: an intermediate representation
// provided to CG backends (LLVM, OpenCL, etc.)

#include <boost/variant.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tile/lang/shape.h"

namespace vertexai {
namespace tile {
namespace sem {

// Simple typing.  Every type is either a TILE value type, a TILE pointer type, or an index
// TILE value types consist of the underlying type and a vector width
struct Type : public el::Loggable {
  enum BaseType { TVOID, INDEX, VALUE, POINTER_MUT, POINTER_CONST };
  enum MemoryRegion { NORMAL, LOCAL, GLOBAL };

  Type(BaseType base_in = TVOID, lang::DataType dtype_in = lang::DataType::INVALID, uint64_t vec_width_in = 1,
       uint64_t array_in = 0, MemoryRegion region_in = NORMAL)
      : base{base_in}, dtype{dtype_in}, vec_width{vec_width_in}, array{array_in}, region{region_in} {}

  void log(el::base::type::ostream_t& os) const final;  // NOLINT(runtime/references)

  BaseType base;
  lang::DataType dtype;
  uint64_t vec_width;  // 1 is non-vector
  uint64_t array;      // 0 is non-array, otherwise, array size
  MemoryRegion region;
};

std::string to_string(const Type& ty);

inline std::ostream& operator<<(::std::ostream& os, const Type& ty) { return os << to_string(ty); }

// Semtrees are trees of various nodes: statements, expressions, lvalues, and functions.
// We first forward-declare the leaf types (leaves of the type tree, not semantic tree leaves), and use them to define
// the variant types that the leaf types point to.

// Statements have side effects and return void.

struct StoreStmt;
struct DeclareStmt;
struct Block;
struct IfStmt;
struct ForStmt;
struct WhileStmt;
struct BarrierStmt;
struct ReturnStmt;

using Statement = boost::variant<StoreStmt, DeclareStmt, Block, IfStmt, ForStmt, WhileStmt, BarrierStmt, ReturnStmt>;
using StmtPtr = std::shared_ptr<Statement>;

// Expressions have no side effects and return a value.

struct IntConst;
struct FloatConst;
struct LoadExpr;
struct UnaryExpr;
struct BinaryExpr;
struct CondExpr;
struct SelectExpr;
struct ClampExpr;
struct CastExpr;
struct CallExpr;
struct LimitConst;
struct IndexExpr;

using Expression = boost::variant<IntConst, FloatConst, LoadExpr, UnaryExpr, BinaryExpr, CondExpr, SelectExpr,
                                  ClampExpr, CastExpr, CallExpr, LimitConst, IndexExpr>;
using ExprPtr = std::shared_ptr<Expression>;

// LValues can be loaded from or stored into, and have no side effects.

struct LookupLVal;
struct SubscriptLVal;

using LValue = boost::variant<LookupLVal, SubscriptLVal>;
using LValPtr = std::shared_ptr<LValue>;

// Now that the variants have been defined, we define the leaf types.

// A constant value.
struct IntConst {
  int64_t value;
  explicit IntConst(int64_t val) : value(val) {}
};

struct FloatConst {
  double value;
  explicit FloatConst(double val) : value(val) {}
};

// A symbol table lookup (ie, a variable name)
struct LookupLVal {
  std::string name;
  explicit LookupLVal(const std::string& n) : name(n) {}
};

// A load from an LVAL (variable or pointer)
struct LoadExpr {
  LValPtr inner;
  explicit LoadExpr(const LValPtr in) : inner(in) {}
};

// A store to an LVAL
struct StoreStmt {
  LValPtr lhs;
  ExprPtr rhs;
  StoreStmt(const LValPtr l, const ExprPtr r) : lhs(std::move(l)), rhs(std::move(r)) {}
};

// Create LVAL reference to an array element
struct SubscriptLVal {
  LValPtr ptr;
  ExprPtr offset;
  SubscriptLVal(const LValPtr p, const ExprPtr o) : ptr(std::move(p)), offset(std::move(o)) {}
};

// A declaration of a new variable (local to scope)
struct DeclareStmt {
  Type type;
  std::string name;
  ExprPtr init;
  DeclareStmt(const Type& t, const std::string n, ExprPtr i) : type(t), name(std::move(n)), init(std::move(i)) {}
};

// A unary operator
struct UnaryExpr {
  std::string op;
  ExprPtr inner;
  UnaryExpr(std::string o, ExprPtr i) : op(std::move(o)), inner(std::move(i)) {}
};

// A binary operator
struct BinaryExpr {
  std::string op;
  ExprPtr lhs;
  ExprPtr rhs;
  BinaryExpr(std::string o, ExprPtr l, ExprPtr r) : op(std::move(o)), lhs(std::move(l)), rhs(std::move(r)) {}
};

// Conditional operator: evaluates either tcase or fcase
struct CondExpr {
  ExprPtr cond;
  ExprPtr tcase;
  ExprPtr fcase;
  CondExpr(ExprPtr c, ExprPtr t, ExprPtr f) : cond(std::move(c)), tcase(std::move(t)), fcase(std::move(f)) {}
};

// Select operator: evaluates both tcase and fcase
struct SelectExpr {
  ExprPtr cond;
  ExprPtr tcase;
  ExprPtr fcase;
  SelectExpr(ExprPtr c, ExprPtr t, ExprPtr f) : cond(std::move(c)), tcase(std::move(t)), fcase(std::move(f)) {}
};

// Clamp operator: constrains a value within limits
struct ClampExpr {
  ExprPtr val;
  ExprPtr min;
  ExprPtr max;
  ClampExpr(ExprPtr v, ExprPtr n, ExprPtr x) : val(std::move(v)), min(std::move(n)), max(std::move(x)) {}
};

// Type conversion equivalent to static_cast
struct CastExpr {
  Type type;
  ExprPtr val;
  CastExpr(const Type& t, ExprPtr v) : type(t), val(std::move(v)) {}
};

// A call of a function
struct CallExpr {
  ExprPtr func;
  std::vector<ExprPtr> vals;
  CallExpr(ExprPtr f, std::vector<ExprPtr> v) : func(std::move(f)), vals(std::move(v)) {}
};

// Represents a type specific constant (min, max, etc)
struct LimitConst {
  enum Which { MIN, MAX, ZERO, ONE };
  Which which;
  lang::DataType type;
  LimitConst(Which _which, const lang::DataType& _type) : which(_which), type(_type) {}
};

// Represents an thread/grid id value
struct IndexExpr {
  enum Type { GLOBAL, GROUP, LOCAL };
  Type type;
  size_t dim;
  IndexExpr(Type _type, size_t _dim) : type(_type), dim(_dim) {}
};

using BlockPtr = std::shared_ptr<Block>;

// A block of statements, also a scope for locals
struct Block {
  std::shared_ptr<std::vector<StmtPtr>> statements;
  Block();
  explicit Block(std::vector<StmtPtr> s);
  explicit Block(StmtPtr s);

  // Adds the supplied block's statements to the current block.
  // This captures the block's statements; further modifications
  // to the block will not appear in the current block.
  void merge(BlockPtr other);

  // Adds the supplied statement to the current block.
  void append(StmtPtr p);
};

// An if clause
struct IfStmt {
  ExprPtr cond;
  BlockPtr iftrue;
  BlockPtr iffalse;
  IfStmt(ExprPtr c, BlockPtr t, BlockPtr f) : cond(c), iftrue(std::move(t)), iffalse(std::move(f)) {}
};

// A highly simplified For statement to allow easier analysis
// Basically, things in the form of:
// for(ssize_t i = 0; i < n*s; i += s) ...
struct ForStmt {
  std::string var;
  uint64_t num;
  uint64_t step;
  BlockPtr inner;
  ForStmt(std::string v, uint64_t n, uint64_t s, BlockPtr i)
      : var(std::move(v)), num(n), step(s), inner(std::move(i)) {}
};

// A while loop
struct WhileStmt {
  ExprPtr cond;
  BlockPtr inner;
  WhileStmt(ExprPtr c, BlockPtr i) : cond(c), inner(std::move(i)) {}
};

// A break statement
struct BreakStmt {};

// A continue statement
struct ContinueStmt {};

// A statement representing an inter-thread barrier
struct BarrierStmt {};

// A return statement
struct ReturnStmt {
  ExprPtr value;
  explicit ReturnStmt(ExprPtr v) : value(std::move(v)) {}
};

// A function.
struct Function {
  typedef std::pair<Type, std::string> param_t;
  typedef std::vector<param_t> params_t;
  std::string name;
  Type ret;
  params_t params;
  BlockPtr body;
  Function() {}
  Function(std::string n, const Type& r, const params_t& p, BlockPtr b)
      : name(std::move(n)), ret(r), params(p), body(std::move(b)) {}
};

}  // namespace sem
}  // namespace tile
}  // namespace vertexai
