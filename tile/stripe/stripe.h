// Copyright 2018, Intel Corporation

#pragma once

#include <functional>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <boost/optional.hpp>

#include "tile/base/shape.h"
#include "tile/math/polynomial.h"
#include "tile/stripe/stripe.pb.h"

namespace vertexai {
namespace tile {
namespace stripe {

using Affine = math::Affine;

enum class StmtKind {
  Load,
  Store,
  Constant,
  LoadIndex,
  Special,
  Intrinsic,
  Block,
};

struct Load;
struct Store;
struct Constant;
struct LoadIndex;
struct Special;
struct Intrinsic;
struct Block;

struct ConstStmtVisitor {
  virtual void Visit(const Load&) = 0;
  virtual void Visit(const Store&) = 0;
  virtual void Visit(const Constant&) = 0;
  virtual void Visit(const LoadIndex&) = 0;
  virtual void Visit(const Special&) = 0;
  virtual void Visit(const Intrinsic&) = 0;
  virtual void Visit(const Block&) = 0;
};

struct MutableStmtVisitor {
  virtual void Visit(Load*) = 0;
  virtual void Visit(Store*) = 0;
  virtual void Visit(Constant*) = 0;
  virtual void Visit(LoadIndex*) = 0;
  virtual void Visit(Special*) = 0;
  virtual void Visit(Intrinsic*) = 0;
  virtual void Visit(Block*) = 0;
};

struct RewriteStmtVisitor {
  virtual Load* Visit(const Load&) = 0;
  virtual Store* Visit(const Store&) = 0;
  virtual Constant* Visit(const Constant&) = 0;
  virtual LoadIndex* Visit(const LoadIndex&) = 0;
  virtual Special* Visit(const Special&) = 0;
  virtual Intrinsic* Visit(const Intrinsic&) = 0;
  virtual Block* Visit(const Block&) = 0;
};

struct Statement;

using StatementList = std::list<std::shared_ptr<Statement>>;
using StatementIt = StatementList::iterator;

using Tags = std::set<std::string>;

class TagVisitor {
 public:
  virtual ~TagVisitor() {}
  virtual void Visit(const std::string& name) = 0;
  virtual void Visit(const std::string& name, bool value) = 0;
  virtual void Visit(const std::string& name, int64_t value) = 0;
  virtual void Visit(const std::string& name, double value) = 0;
  virtual void Visit(const std::string& name, const std::string& value) = 0;
  virtual void Visit(const std::string& name, const google::protobuf::Any& value) = 0;
};

// Generic properties used by optimization passes
class Taggable {
  friend struct Accessor;

 protected:
  Taggable();

 public:
  // Copy constructor
  Taggable(const Taggable& rhs);

  // Copy assignment
  Taggable& operator=(const Taggable& rhs);

  ~Taggable();

  void set_tag(const std::string& tag);
  void set_tags(const Tags& tags);
  void add_tags(const Tags& to_add);
  void clear_tags();
  void remove_tag(const std::string& tag);
  void remove_tags(const Tags& tags);

  bool has_tag(const std::string& tag) const;
  bool has_tags(const Tags& to_find) const;
  bool has_any_tags(const Tags& to_find) const;

  bool any_tags() const;
  void visit_tags(TagVisitor* visitor) const;

  void set_attr(const std::string& name);
  void set_attr(const std::string& name, bool value);
  void set_attr(const std::string& name, int64_t value);
  void set_attr(const std::string& name, double value);
  void set_attr(const std::string& name, const std::string& value);
  void set_attr(const std::string& name, const google::protobuf::Any& value);
  void set_attrs(const Taggable& rhs);

  bool has_attr(const std::string& name) const;
  bool get_attr_bool(const std::string& name) const;
  int64_t get_attr_int(const std::string& name) const;
  double get_attr_float(const std::string& name) const;
  std::string get_attr_str(const std::string& name) const;
  google::protobuf::Any get_attr_any(const std::string& name) const;

  bool get_attr_bool(const std::string& name, bool def) const;
  int64_t get_attr_int(const std::string& name, int64_t def) const;
  double get_attr_float(const std::string& name, double def) const;
  std::string get_attr_str(const std::string& name, const std::string& def) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class Codec {
 public:
  using Factory = std::function<std::unique_ptr<Codec>(const TensorShape* shape)>;

  static void Register(const std::string& name, const Factory& factory);
  static std::unique_ptr<Codec> Resolve(const TensorShape& shape);

  virtual ~Codec() = default;
  virtual int64_t byte_size() const = 0;
  virtual boost::optional<size_t> sparse_dim() const = 0;

 protected:
  explicit Codec(const TensorShape* shape);
  const TensorShape* shape_;
};

struct Statement : Taggable {
  virtual ~Statement() = default;
  virtual StmtKind kind() const = 0;
  virtual std::vector<std::string> buffer_reads() const { return {}; }
  virtual std::vector<std::string> buffer_writes() const { return {}; }
  virtual std::vector<std::string> scalar_uses() const { return {}; }
  virtual std::vector<std::string> scalar_defs() const { return {}; }
  virtual void Accept(ConstStmtVisitor*) const = 0;
  virtual void Accept(MutableStmtVisitor*) = 0;
  virtual Statement* Accept(RewriteStmtVisitor*) = 0;

  // The set of statements within the same Block that must complete
  // before this statement is evaluated.
  std::list<StatementIt> deps;
};

struct Index : Taggable {
  Index(const std::string& name,  //
        uint64_t range,           //
        const Affine& affine = Affine{})
      : name(name),  //
        range(range),
        affine(affine) {}
  std::string name;
  uint64_t range;
  Affine affine;
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

struct Device {
  std::string name;
  std::vector<Affine> units;
};

struct Location {
  std::vector<Device> devs;

  bool empty() const { return devs.size() == 0; }
};

struct BankDimension {
  size_t dim_pos;
};

struct Refinement : Taggable {
  Refinement() = default;
  Refinement(RefDir dir,                             //
             const std::string& from,                //
             const std::string& into,                //
             const std::vector<Affine>& access,      //
             const TensorShape& interior_shape,      //
             const std::string& agg_op = "",         //
             const Location& location = Location{},  //
             uint64_t offset = 0,                    //
             const boost::optional<BankDimension>& bank_dim = boost::none,
             const boost::optional<Affine>& cache_unit = boost::none)
      : dir(dir),
        from(from),
        access(access),
        interior_shape(interior_shape),
        agg_op(agg_op),
        location(location),
        offset(offset),
        bank_dim(bank_dim),
        cache_unit(cache_unit),
        into_{into} {}

  Refinement WithInto(const std::string& into) const {
    Refinement result = *this;
    result.into_ = into;
    return result;
  }

  static Refinement FromInto(const std::string& into) {
    Refinement result;
    result.into_ = into;
    return result;
  }

  const std::string& into() const { return into_; }

  RefDir dir = RefDir::None;
  std::string from;
  std::vector<Affine> access;
  TensorShape interior_shape;
  std::string agg_op;
  Location location;
  uint64_t offset = 0;                      // Offset within the location's arena.
  boost::optional<BankDimension> bank_dim;  // Which dimension should we bank on
  boost::optional<Affine> cache_unit;       // Which cache we should use when encaching this refinement

  Affine FlatAccess() const;
  TensorShape ApplyTile(const std::map<std::string, size_t>& tile_by_name) const;

  // Returns a mutable Refinement from a const Refinement.  This is
  // useful when processing a set<Refinement>, since for safety
  // reasons, the normal accessors only return access to const
  // Refinements.
  Refinement& mut() const { return const_cast<Refinement&>(*this); }

 private:
  std::string into_;
};

}  // namespace stripe
}  // namespace tile
}  // namespace vertexai

// The default comparator for Refinements is to compare the "into"
// field.  We do this by specializing std::less<Refinement> so that we
// can add comparators for strings.
namespace std {

template <>
struct less<::vertexai::tile::stripe::Refinement> {
  using is_transparent = void;  // Allows string comparators
  const bool operator()(const ::vertexai::tile::stripe::Refinement& lhs,
                        const ::vertexai::tile::stripe::Refinement& rhs) const {
    return std::less<std::string>{}(lhs.into(), rhs.into());
  }
  const bool operator()(const ::vertexai::tile::stripe::Refinement& lhs, const std::string& rhs) const {
    return std::less<std::string>{}(lhs.into(), rhs);
  }
  const bool operator()(const ::vertexai::tile::stripe::Refinement& lhs, const char* rhs) const {
    return std::less<std::string>{}(lhs.into(), rhs);
  }
  const bool operator()(const std::string& lhs, const ::vertexai::tile::stripe::Refinement& rhs) const {
    return std::less<std::string>{}(lhs, rhs.into());
  }
  const bool operator()(const char* lhs, const ::vertexai::tile::stripe::Refinement& rhs) const {
    return std::less<std::string>{}(lhs, rhs.into());
  }
};

}  // namespace std

namespace vertexai {
namespace tile {
namespace stripe {

struct Load : Statement {
  Load(const std::string& from, const std::string& into) : from(from), into(into) {}
  static std::shared_ptr<Load> Downcast(const std::shared_ptr<Statement>& stmt);
  StmtKind kind() const { return StmtKind::Load; }
  std::vector<std::string> buffer_reads() const { return {from}; }
  std::vector<std::string> scalar_defs() const { return {into}; }
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
  std::vector<std::string> buffer_writes() const { return {into}; }
  std::vector<std::string> scalar_uses() const { return {from}; }
  void Accept(ConstStmtVisitor* v) const { v->Visit(*this); }
  void Accept(MutableStmtVisitor* v) { v->Visit(this); }
  Store* Accept(RewriteStmtVisitor* v) { return v->Visit(*this); }

  std::string from;
  std::string into;
};

struct LoadIndex : Statement {
  LoadIndex(const Affine& from, const std::string& into) : from(from), into(into) {}
  static std::shared_ptr<LoadIndex> Downcast(const std::shared_ptr<Statement>& stmt);
  StmtKind kind() const { return StmtKind::LoadIndex; }
  std::vector<std::string> scalar_defs() const { return {into}; }
  void Accept(ConstStmtVisitor* v) const { v->Visit(*this); }
  void Accept(MutableStmtVisitor* v) { v->Visit(this); }
  LoadIndex* Accept(RewriteStmtVisitor* v) { return v->Visit(*this); }

  Affine from;
  std::string into;
};

struct Intrinsic : Statement {
  static std::shared_ptr<Intrinsic> Downcast(const std::shared_ptr<Statement>& stmt);
  StmtKind kind() const { return StmtKind::Intrinsic; }
  std::vector<std::string> scalar_uses() const { return inputs; }
  std::vector<std::string> scalar_defs() const { return outputs; }
  void Accept(ConstStmtVisitor* v) const { v->Visit(*this); }
  void Accept(MutableStmtVisitor* v) { v->Visit(this); }
  Intrinsic* Accept(RewriteStmtVisitor* v) { return v->Visit(*this); }

  std::string name;
  DataType type = DataType::FLOAT32;
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
  std::vector<std::string> buffer_reads() const { return inputs; }
  std::vector<std::string> buffer_writes() const { return outputs; }
  void Accept(ConstStmtVisitor* v) const { v->Visit(*this); }
  void Accept(MutableStmtVisitor* v) { v->Visit(this); }
  Special* Accept(RewriteStmtVisitor* v) { return v->Visit(*this); }

  std::string name;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::map<std::string, int64_t> int_params;
  std::map<std::string, std::string> str_params;
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
  std::vector<std::string> scalar_defs() const { return {name}; }
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
  std::vector<std::string> buffer_reads() const;
  std::vector<std::string> buffer_writes() const;
  void Accept(ConstStmtVisitor* v) const { v->Visit(*this); }
  void Accept(MutableStmtVisitor* v) { v->Visit(this); }
  Block* Accept(RewriteStmtVisitor* v) { return v->Visit(*this); }

  std::string name;
  std::string comments;
  std::vector<Index> idxs;
  std::vector<Affine> constraints;
  std::set<Refinement> refs;
  StatementList stmts;
  Location location;

  // Helper methods
  std::vector<const Refinement*> ref_ins(bool inout = false) const;
  std::vector<const Refinement*> ref_outs(bool inout = false) const;
  std::vector<const Refinement*> ref_inouts() const;
  std::vector<Refinement*> ref_ins(bool inout = false);
  std::vector<Refinement*> ref_outs(bool inout = false);
  std::vector<Refinement*> ref_inouts();
  Index* idx_by_name(const std::string& name);
  const Index* idx_by_name(const std::string& name) const;
  std::set<const Index*> accumulation_idxs(bool inout = false) const;
  size_t idxs_product() const;
  // Find which refinement has an into called 'ref_name'
  std::set<Refinement>::iterator ref_by_into(const std::string& ref_name, bool fail = true);
  std::set<Refinement>::const_iterator ref_by_into(const std::string& ref_name, bool fail = true) const;
  // Find which refinement has a from called 'ref_name'
  std::set<Refinement>::iterator ref_by_from(const std::string& ref_name, bool fail = true);
  std::set<Refinement>::const_iterator ref_by_from(const std::string& ref_name, bool fail = true) const;
  // Find which refinement has a tag called 'tag_name'
  std::set<Refinement>::iterator ref_by_tag(const std::string& tag_name, bool fail = true);
  std::set<Refinement>::const_iterator ref_by_tag(const std::string& tag_name, bool fail = true) const;
  // Make a unique refinement name for an into (by appending _2, etc, if needed)
  std::string unique_ref_name(const std::string& into) const;
  // Make a unique index name (by appending _2, etc, if needed)
  std::string unique_idx_name(const std::string& name) const;
  TensorShape exterior_shape(const std::string& name) const;
  // Sorted index ranges
  std::vector<size_t> sorted_idx_ranges();

  std::shared_ptr<Block> SubBlock(size_t pos, bool reverse = false) const;
  void erase_stmt(const StatementIt& it);
  void erase_stmts(const StatementIt& begin, const StatementIt& end);
};

struct Buffer {
  std::map<std::string, std::string> sections;
};

struct Program {
  std::map<std::string, Buffer> buffers;
  std::shared_ptr<Block> entry;
  ShapeMap input_shapes;
  ShapeMap output_shapes;
};

std::string to_string(RefDir dir);

inline bool operator<(const StatementIt& lhs, const StatementIt& rhs) {  //
  return lhs->get() < rhs->get();
}

bool operator==(const BankDimension& lhs, const BankDimension& rhs);
bool operator==(const Index& lhs, const Index& rhs);

bool operator==(const Device& lhs, const Device& rhs);
bool operator!=(const Device& lhs, const Device& rhs);
bool operator<(const Device& lhs, const Device& rhs);

std::string to_string(const Device& dev);

// Applies the supplied parameter map to the device's units.
Device PartialEval(const Device& dev, const std::map<std::string, std::int64_t>& values);

bool operator==(const Location& lhs, const Location& rhs);
bool operator!=(const Location& lhs, const Location& rhs);
bool operator<(const Location& lhs, const Location& rhs);

// Adds the location's units, returning the result.  The locations
// should have the same shape (device names and unit dimensionality).
Location AddDeviceUnits(const Location& loc1, const Location& loc2);

// Appends two locations, returning the result.
Location AppendLocations(const Location& loc1, const Location& loc2);

// Replaces tags in a location with concrete index references (using
// the first matching index).
Location RealizeLocation(const Location& loc, const Block& block);

std::string to_string(const Location& loc);

// Returns a match if the location matches the indicated pattern,
// which looks like a slash-separated string of device names with unit
// vectors.
//
// '*' matches any device or unit; otherwise, matches must be exact.
// When '*' is used to match a device, a unit may not be specified.
//
// If the unit vector is omitted, any number of units matches.
//
// For example: "d1/*/d3[1,*]" would match "d1[4]/d2[1,2]/d3[1,6]",
// since the names and wildcards match up.
//
// Note that this function does not return the matched parameters (the
// names or affines); it's akin to a typecheck, in that the caller may
// subsequently safely depend on the contents of the location.
bool operator==(const Location& loc, const std::string& pattern);
inline bool operator!=(const Location& loc, const std::string& pattern) { return !(loc == pattern); }

// Applies the supplied parameter map to the location's units.
Location PartialEval(const Location& loc, const std::map<std::string, std::int64_t>& values);

struct PrintRefinement {
  explicit PrintRefinement(const Refinement& ref, const Block* block = nullptr) : ref(ref), block(block) {}

  const Refinement& ref;
  const Block* block = nullptr;
};

std::ostream& operator<<(std::ostream& os, const Device& dev);
std::ostream& operator<<(std::ostream& os, const Location& loc);
std::ostream& operator<<(std::ostream& os, const Index& idx);
std::ostream& operator<<(std::ostream& os, const Load& op);
std::ostream& operator<<(std::ostream& os, const Store& op);
std::ostream& operator<<(std::ostream& os, const LoadIndex& op);
std::ostream& operator<<(std::ostream& os, const Intrinsic& op);
std::ostream& operator<<(std::ostream& os, const Special& op);
std::ostream& operator<<(std::ostream& os, const Constant& op);
std::ostream& operator<<(std::ostream& os, const Refinement& ref);
std::ostream& operator<<(std::ostream& os, const PrintRefinement& ref);
std::ostream& operator<<(std::ostream& os, const Block& block);

bool FromProtoText(const std::string& pbtxt, proto::Program* into);
std::shared_ptr<Program> FromProto(const proto::Program& program);
std::shared_ptr<Block> FromProto(const proto::Block& block);
Affine FromProto(const proto::Affine& affine);
Device FromProto(const proto::Device& dev);
std::vector<Device> FromProto(const google::protobuf::RepeatedPtrField<proto::Device>& devs);
Location FromProto(const proto::Location& loc);
RefDir FromProto(const proto::Refinement::Dir& dir);
Tags FromProto(const google::protobuf::RepeatedPtrField<std::string>& pb_tags);

proto::Block IntoProto(const Block& block);
proto::Program IntoProto(const Program& program);

std::shared_ptr<Block> CloneBlock(const Block& orig, int depth = -1);
const Block* FindBlockByTag(const Block& block, const std::string& tag);
void FindBlocksByTag(std::vector<const Block*>* into, const Block& block, const std::string& tag);
const Index* FindIndexByTag(const Block& block, const std::string& tag);

bool InsertAfterBlock(Block* parent, Block* sub, std::shared_ptr<Statement> stmt);

template <typename F>
void PreIterate(Block* block, const F& func) {
  auto it = block->stmts.begin();
  while (it != block->stmts.end()) {
    auto next = it;
    ++next;
    func(it);
    it = next;
  }
}

inline std::string to_string(const Block& block) {
  std::stringstream ss;
  ss << block;
  return ss.str();
}

}  // namespace stripe

namespace math {

inline std::ostream& operator<<(std::ostream& os, const stripe::Affine& affine) {
  os << affine.toString();
  return os;
}

}  // namespace math

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
