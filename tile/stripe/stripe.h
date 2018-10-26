#pragma once

#include <functional>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tile/base/shape.h"
#include "tile/math/polynomial.h"
#include "tile/stripe/stripe.pb.h"

namespace vertexai {
namespace tile {
namespace stripe {

using Affine = math::Polynomial<int64_t>;

enum class StmtKind {
  Load,
  Store,
  Constant,
  Special,
  Intrinsic,
  Block,
};

struct Load;
struct Store;
struct Constant;
struct Special;
struct Intrinsic;
struct Block;

class ConstStmtVisitor {
 public:
  virtual void Visit(const Load&) = 0;
  virtual void Visit(const Store&) = 0;
  virtual void Visit(const Constant&) = 0;
  virtual void Visit(const Special&) = 0;
  virtual void Visit(const Intrinsic&) = 0;
  virtual void Visit(const Block&) = 0;
};

class MutableStmtVisitor {
 public:
  virtual void Visit(Load*) = 0;
  virtual void Visit(Store*) = 0;
  virtual void Visit(Constant*) = 0;
  virtual void Visit(Special*) = 0;
  virtual void Visit(Intrinsic*) = 0;
  virtual void Visit(Block*) = 0;
};

class RewriteStmtVisitor {
 public:
  virtual Load* Visit(const Load&) = 0;
  virtual Store* Visit(const Store&) = 0;
  virtual Constant* Visit(const Constant&) = 0;
  virtual Special* Visit(const Special&) = 0;
  virtual Intrinsic* Visit(const Intrinsic&) = 0;
  virtual Block* Visit(const Block&) = 0;
};

struct Statement;

using StatementList = std::list<std::shared_ptr<Statement>>;
using StatementIt = StatementList::iterator;

inline bool operator<(const StatementIt& lhs, const StatementIt& rhs) { return lhs->get() < rhs->get(); }

struct Statement {
  virtual ~Statement() = default;
  virtual StmtKind kind() const = 0;
  virtual void Accept(ConstStmtVisitor*) const = 0;
  virtual void Accept(MutableStmtVisitor*) = 0;
  virtual Statement* Accept(RewriteStmtVisitor*) = 0;

  // The set of statements within the same Block that must complete
  // before this statement is evaluated.
  std::list<StatementIt> deps;
};

struct Annotation {
  virtual ~Annotation() = default;
};

struct Index {
  Index() : range(0), factor(0) {}
  Index(const std::string& name, const std::string& from, uint64_t range, int64_t factor)
      : name(name), from(from), range(range), factor(factor) {}

  std::string name;
  std::string from;
  uint64_t range;
  int64_t factor;
};

enum class RefDir {
  None = 0,
  In = 1,
  Out = 2,
  InOut = 3,
};

inline bool IsReadDir(const RefDir& dir) {  //
  return dir == RefDir::In || dir == RefDir::InOut;
}

inline bool IsWriteDir(const RefDir& dir) {  //
  return dir == RefDir::Out || dir == RefDir::InOut;
}

inline RefDir UnionDir(const RefDir& a, const RefDir& b) {  //
  return RefDir(static_cast<int>(a) | static_cast<int>(b));
}

struct Location {
  std::string name;
  Affine unit;
};

struct Refinement {
  RefDir dir;
  std::string from;
  std::string into;
  std::vector<Affine> access;
  TensorShape shape;
  std::string agg_op;
  Location location;
  bool is_const;

  Affine FlatAccess() const;
};

struct Load : Statement {
  Load(const std::string& from, const std::string& into) : from(from), into(into) {}
  static std::shared_ptr<Load> Downcast(const std::shared_ptr<Statement>& stmt);
  StmtKind kind() const { return StmtKind::Load; }
  void Accept(ConstStmtVisitor* v) const { v->Visit(*this); }
  void Accept(MutableStmtVisitor* v) { v->Visit(this); }
  Load* Accept(RewriteStmtVisitor* v) { return v->Visit(*this); }

  std::string from;
  std::string into;
};

struct Store : Statement {
  Store(const std::string& from, const std::string& into) : from(from), into(into) {}
  static std::shared_ptr<Store> Downcast(const std::shared_ptr<Statement>& stmt);
  StmtKind kind() const { return StmtKind::Store; }
  void Accept(ConstStmtVisitor* v) const { v->Visit(*this); }
  void Accept(MutableStmtVisitor* v) { v->Visit(this); }
  Store* Accept(RewriteStmtVisitor* v) { return v->Visit(*this); }

  std::string from;
  std::string into;
};

struct Intrinsic : Statement {
  static std::shared_ptr<Intrinsic> Downcast(const std::shared_ptr<Statement>& stmt);
  StmtKind kind() const { return StmtKind::Intrinsic; }
  void Accept(ConstStmtVisitor* v) const { v->Visit(*this); }
  void Accept(MutableStmtVisitor* v) { v->Visit(this); }
  Intrinsic* Accept(RewriteStmtVisitor* v) { return v->Visit(*this); }

  std::string name;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;

  static const char* ASSIGN;
  static const char* SUM;
  static const char* MIN;
  static const char* MAX;
  static const char* PROD;

  static const char* MUL;
  static const char* ADD;
  static const char* EQ;
  static const char* COND;
};

struct Special : Statement {
  static std::shared_ptr<Special> Downcast(const std::shared_ptr<Statement>& stmt);
  StmtKind kind() const { return StmtKind::Special; }
  void Accept(ConstStmtVisitor* v) const { v->Visit(*this); }
  void Accept(MutableStmtVisitor* v) { v->Visit(this); }
  Special* Accept(RewriteStmtVisitor* v) { return v->Visit(*this); }

  std::string name;
  std::vector<std::string> params;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;

  static const char* ZERO;
  static const char* COPY;
};

enum class ConstType {
  Integer,
  Float,
};

struct Constant : Statement {
  Constant(const std::string& name, int64_t value) : name(name), type(ConstType::Integer), iconst(value) {}
  Constant(const std::string& name, double value) : name(name), type(ConstType::Float), fconst(value) {}
  static std::shared_ptr<Constant> Downcast(const std::shared_ptr<Statement>& stmt);
  StmtKind kind() const { return StmtKind::Constant; }
  void Accept(ConstStmtVisitor* v) const { v->Visit(*this); }
  void Accept(MutableStmtVisitor* v) { v->Visit(this); }
  Constant* Accept(RewriteStmtVisitor* v) { return v->Visit(*this); }

  std::string name;
  ConstType type;
  int64_t iconst;
  double fconst;
};

struct Block : Statement {
  static std::shared_ptr<Block> Downcast(const std::shared_ptr<Statement>& stmt);
  StmtKind kind() const { return StmtKind::Block; }
  void Accept(ConstStmtVisitor* v) const { v->Visit(*this); }
  void Accept(MutableStmtVisitor* v) { v->Visit(this); }
  Block* Accept(RewriteStmtVisitor* v) { return v->Visit(*this); }

  std::string name;
  std::string comments;
  std::vector<Index> idxs;
  std::vector<Affine> constraints;
  std::vector<Refinement> refs;
  StatementList stmts;
  std::map<std::string, std::shared_ptr<Annotation>> annotations;
  Location location;

  // Helper methods
  std::vector<const Refinement*> ref_ins() const;
  std::vector<const Refinement*> ref_outs() const;
  const Index* idx_by_name(const std::string& name) const;
  // Find which refinement has an into called 'name'
  std::vector<Refinement>::iterator ref_by_into(const std::string& name);
  std::vector<Refinement>::const_iterator ref_by_into(const std::string& name) const;
  // Find which refinement has a from called 'name'
  std::vector<Refinement>::iterator ref_by_from(const std::string& name);
  std::vector<Refinement>::const_iterator ref_by_from(const std::string& name) const;
  // Make a unique refinement name for an into (by appending _2, etc, if needed)
  std::string unique_ref_name(const std::string& in);
};

struct BoolAnnotation : Annotation {
  explicit BoolAnnotation(bool value) : value(value) {}
  static std::shared_ptr<BoolAnnotation> Downcast(const std::shared_ptr<Annotation>& ann);
  bool value;
};

bool operator==(const Index& lhs, const Index& rhs);

std::ostream& operator<<(std::ostream& os, const Location& loc);
std::ostream& operator<<(std::ostream& os, const Index& idx);
std::ostream& operator<<(std::ostream& os, const Load& op);
std::ostream& operator<<(std::ostream& os, const Store& op);
std::ostream& operator<<(std::ostream& os, const Intrinsic& op);
std::ostream& operator<<(std::ostream& os, const Special& op);
std::ostream& operator<<(std::ostream& os, const Constant& op);
std::ostream& operator<<(std::ostream& os, const Block& block);

std::shared_ptr<Block> FromProto(const proto::Block& block);
proto::Block IntoProto(const Block& block);

class CloneVisitor : RewriteStmtVisitor {
 public:
  explicit CloneVisitor(size_t depth) : depth_(depth) {}
  Load* Visit(const Load& x) { return new Load(x); }
  Store* Visit(const Store& x) { return new Store(x); }
  Constant* Visit(const Constant& x) { return new Constant(x); }
  Special* Visit(const Special& x) { return new Special(x); }
  Intrinsic* Visit(const Intrinsic& x) { return new Intrinsic(x); }
  Block* Visit(const Block& x) {
    auto ret = new Block(x);
    if (depth_ == 0) {
      return ret;
    }
    depth_--;
    for (auto& stmt_ptr : ret->stmts) {
      stmt_ptr = std::shared_ptr<Statement>(stmt_ptr->Accept(this));
    }
    depth_++;
    return ret;
  }

 private:
  size_t depth_;
};

inline std::shared_ptr<Block> CloneBlock(const Block& orig, size_t depth = -1) {
  CloneVisitor visitor(depth);
  return std::shared_ptr<Block>(visitor.Visit(orig));
}

}  // namespace stripe
}  // namespace tile
}  // namespace vertexai

namespace std {

template <>
struct hash<vertexai::tile::stripe::StatementIt> {
  std::size_t operator()(const vertexai::tile::stripe::StatementIt& it) const {
    return hash<vertexai::tile::stripe::Statement*>{}(it->get());
  }
};

}  // namespace std
