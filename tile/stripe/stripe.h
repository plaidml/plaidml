#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "tile/base/shape.h"
#include "tile/stripe/stripe.pb.h"

namespace vertexai {
namespace tile {
namespace stripe {

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
  std::string name;
  uint64_t range;
  int64_t factor;
};

enum class RefDir {
  In,
  Out,
  InOut,
};

struct BufferAccess {
  int64_t offset;
  std::vector<int64_t> strides;
};

struct Refinement {
  RefDir dir;
  std::string from;
  std::string into;
  BufferAccess access;
  TensorShape shape;
  std::string agg_op;
};

struct Constraint {
  std::vector<int64_t> lhs;
  int64_t rhs;
};

struct Load : Statement {
  Load(const std::string& from, const std::string& into) : from(from), into(into) {}
  StmtKind kind() const { return StmtKind::Load; }

  std::string from;
  std::string into;
};

struct Store : Statement {
  Store(const std::string& from, const std::string& into) : from(from), into(into) {}
  StmtKind kind() const { return StmtKind::Store; }

  std::string from;
  std::string into;
};

struct Intrinsic : Statement {
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
  StmtKind kind() const { return StmtKind::Constant; }

  std::string name;
  ConstType type;
  int64_t iconst;
  double fconst;
};

struct Block : Statement {
  StmtKind kind() const { return StmtKind::Block; }
  std::vector<Refinement> ref_ins() const;
  std::vector<Refinement> ref_outs() const;

  std::string name;
  std::string comments;
  std::vector<Index> idxs;
  std::vector<Constraint> constraints;
  std::map<std::string, TensorShape> decls;
  std::vector<Refinement> refs;
  std::vector<std::shared_ptr<Statement>> stmts;
  std::map<std::string, std::shared_ptr<Annotation>> annotations;
};

struct BoolAnnotation : Annotation {
  explicit BoolAnnotation(bool value) : value(value) {}
  bool value;
};

bool operator==(const Index& lhs, const Index& rhs);
bool operator==(const Constraint& lhs, const Constraint& rhs);
bool operator==(const BufferAccess& lhs, const BufferAccess& rhs);

std::ostream& operator<<(std::ostream& os, const Index& idx);
std::ostream& operator<<(std::ostream& os, const Block& block);
std::ostream& operator<<(std::ostream& os, const BufferAccess& a);

Block FromProto(const proto::Block& block);
proto::Block IntoProto(const Block& block);

}  // namespace stripe
}  // namespace tile
}  // namespace vertexai
