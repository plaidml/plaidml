#pragma once

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

struct Statement {
  virtual ~Statement() = default;
  virtual StmtKind kind() const = 0;
};

struct Annotation {
  virtual ~Annotation() = default;
};

struct Index {
  Index() : range(0), factor(0) {}
  Index(const std::string& name, uint64_t range, int64_t factor) : name(name), range(range), factor(factor) {}

  std::string name;
  uint64_t range;
  int64_t factor;
};

enum class RefDir {
  None,
  In,
  Out,
  InOut,
};

struct Refinement {
  RefDir dir;
  std::string from;
  std::string into;
  std::vector<Affine> access;
  TensorShape shape;
  std::string agg_op;

  Affine FlatAccess() const;
};

struct Load : Statement {
  Load(const std::string& from, const std::string& into) : from(from), into(into) {}
  static std::shared_ptr<Load> Downcast(const std::shared_ptr<Statement>& stmt);
  StmtKind kind() const { return StmtKind::Load; }

  std::string from;
  std::string into;
};

struct Store : Statement {
  Store(const std::string& from, const std::string& into) : from(from), into(into) {}
  static std::shared_ptr<Store> Downcast(const std::shared_ptr<Statement>& stmt);
  StmtKind kind() const { return StmtKind::Store; }

  std::string from;
  std::string into;
};

struct Intrinsic : Statement {
  static std::shared_ptr<Intrinsic> Downcast(const std::shared_ptr<Statement>& stmt);
  StmtKind kind() const { return StmtKind::Intrinsic; }

  std::string name;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;

  static const char* ZERO;
  static const char* COPY;

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

  std::string name;
  std::vector<std::string> params;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
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

  std::string name;
  ConstType type;
  int64_t iconst;
  double fconst;
};

using StatementList = std::list<std::shared_ptr<Statement>>;
using StatementIt = StatementList::iterator;

struct Block : Statement {
  static std::shared_ptr<Block> Downcast(const std::shared_ptr<Statement>& stmt);
  StmtKind kind() const { return StmtKind::Block; }

  std::string name;
  std::string comments;
  std::vector<Index> idxs;
  std::vector<Affine> constraints;
  std::vector<Refinement> refs;
  StatementList stmts;
  std::map<std::string, std::shared_ptr<Annotation>> annotations;

  // Helper methods
  std::vector<const Refinement*> ref_ins() const;
  std::vector<const Refinement*> ref_outs() const;
  const Index* idx_by_name(const std::string& name) const;
  // Find which refinement has an into called 'name'
  std::vector<Refinement>::iterator ref_by_into(const std::string& name);
  std::vector<Refinement>::const_iterator ref_by_into(const std::string& name) const;
  // Make a unique refinement name for an into (by appending _2, etc, if needed)
  std::string unique_ref_name(const std::string& in);
};

struct BoolAnnotation : Annotation {
  explicit BoolAnnotation(bool value) : value(value) {}
  static std::shared_ptr<BoolAnnotation> Downcast(const std::shared_ptr<Annotation>& ann);
  bool value;
};

bool operator==(const Index& lhs, const Index& rhs);

std::ostream& operator<<(std::ostream& os, const Index& idx);
std::ostream& operator<<(std::ostream& os, const Load& op);
std::ostream& operator<<(std::ostream& os, const Store& op);
std::ostream& operator<<(std::ostream& os, const Intrinsic& op);
std::ostream& operator<<(std::ostream& os, const Special& op);
std::ostream& operator<<(std::ostream& os, const Constant& op);
std::ostream& operator<<(std::ostream& os, const Block& block);

std::shared_ptr<Block> FromProto(const proto::Block& block);
proto::Block IntoProto(const Block& block);

}  // namespace stripe
}  // namespace tile
}  // namespace vertexai
