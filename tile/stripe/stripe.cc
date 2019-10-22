// Copyright 2018, Intel Corporation

#include "tile/stripe/stripe.h"

#include <algorithm>
#include <regex>
#include <sstream>
#include <utility>

#include "boost/format.hpp"
#include "google/protobuf/text_format.h"

#include "base/util/stream_container.h"
#include "base/util/throw.h"
#include "tile/stripe/impl.h"

namespace vertexai {
namespace tile {
namespace stripe {

using google::protobuf::Any;

const char* Intrinsic::ASSIGN = "assign";
const char* Intrinsic::SUM = "add";
const char* Intrinsic::MIN = "min";
const char* Intrinsic::MAX = "max";
const char* Intrinsic::PROD = "mul";

const char* Intrinsic::MUL = "mul";
const char* Intrinsic::ADD = "add";
const char* Intrinsic::EQ = "cmp_eq";
const char* Intrinsic::COND = "cond";

const Taggable::Impl* Accessor::impl(const Taggable& taggable) { return taggable.impl_.get(); }

namespace {

using DepsMap = std::unordered_map<const Statement*, size_t>;

void PrintStmt(std::ostream& os,       //
               const Statement& stmt,  //
               size_t depth,           //
               size_t idx,             //
               const DepsMap& deps);

void PrintTab(std::ostream& os, size_t depth) {  //
  os << std::string(depth * 2, ' ');
}

void PrintPreStmt(std::ostream& os,       //
                  size_t depth,           //
                  const Statement& stmt,  //
                  size_t idx,             //
                  const DepsMap& deps) {
  PrintTab(os, depth);
  os << idx;
  if (stmt.deps.size()) {
    os << "[";
    bool first = true;
    for (const auto& it : stmt.deps) {
      if (first) {
        first = false;
      } else {
        os << ", ";
      }
      auto dep_idx_it = deps.find(it->get());
      if (dep_idx_it != deps.end()) {
        os << dep_idx_it->second;
      } else {
        os << "parent";
      }
    }
    os << "]";
  }
  os << ": ";
  auto impl = Accessor::impl(stmt);
  if (impl->attrs.size()) {
    for (const auto& attr : impl->attrs) {
      os << "#";
      if (attr.second.type() == typeid(Void)) {
        os << attr.first;
      } else {
        os << attr.first << "=" << attr.second;
      }
      os << " ";
    }
    os << std::endl;
    PrintTab(os, depth);
  }
}

void PrintRefinements(std::ostream& os, const Block& block, size_t depth) {
  if (block.refs.size() > 2) {
    std::map<std::string, const Refinement*> sorted;
    for (const auto& ref : block.refs) {
      sorted.emplace(ref.into(), &ref);
    }
    for (const auto& kvp : sorted) {
      PrintTab(os, depth + 2);
      os << PrintRefinement{*kvp.second, &block} << std::endl;
    }
  } else {
    for (const auto& ref : block.refs) {
      PrintTab(os, depth + 2);
      os << PrintRefinement{ref, &block} << std::endl;
    }
  }
}

void PrintBlock(std::ostream& os,    //
                const Block& block,  //
                size_t depth,        //
                size_t block_idx,    //
                const DepsMap& block_deps) {
  os << "block";
  if (!block.location.empty()) {
    os << "<" << block.location << ">";
  }
  os << " [";
  for (size_t i = 0; i < block.idxs.size(); i++) {
    if (i > 0) {
      os << ", ";
    }
    os << block.idxs[i];
  }
  os << "]:" << block.idxs_product() << " (";
  if (!block.name.empty()) {
    os << " // " << block.name;
  }
  os << std::endl;

  if (!block.comments.empty()) {
    std::stringstream ss(block.comments);
    for (std::string line; std::getline(ss, line, '\n');) {
      PrintTab(os, depth + 2);
      os << "// " << line << std::endl;
    }
  }
  for (const auto& constraint : block.constraints) {
    PrintTab(os, depth + 2);
    os << constraint.toString() << " >= 0";
    os << std::endl;
  }
  PrintRefinements(os, block, depth);
  PrintTab(os, depth);
  os << ") {" << std::endl;
  std::size_t idx = 0;
  DepsMap deps;
  for (const auto& stmt : block.stmts) {
    PrintStmt(os, *stmt, depth + 1, idx, deps);
    deps[stmt.get()] = idx++;
  }
  PrintTab(os, depth);
  os << "}" << std::endl;
}

void PrintStmt(std::ostream& os,       //
               const Statement& stmt,  //
               size_t depth,           //
               size_t idx,             //
               const DepsMap& deps) {
  PrintPreStmt(os, depth, stmt, idx, deps);
  switch (stmt.kind()) {
    case StmtKind::Load:
      os << *dynamic_cast<const Load*>(&stmt) << std::endl;
      break;
    case StmtKind::Store:
      os << *dynamic_cast<const Store*>(&stmt) << std::endl;
      break;
    case StmtKind::LoadIndex:
      os << *dynamic_cast<const LoadIndex*>(&stmt) << std::endl;
      break;
    case StmtKind::Intrinsic:
      os << *dynamic_cast<const Intrinsic*>(&stmt) << std::endl;
      break;
    case StmtKind::Special:
      os << *dynamic_cast<const Special*>(&stmt) << std::endl;
      break;
    case StmtKind::Constant:
      os << *dynamic_cast<const Constant*>(&stmt) << std::endl;
      break;
    case StmtKind::Block:
      PrintBlock(os, *dynamic_cast<const Block*>(&stmt), depth, idx, deps);
      break;
    default:
      break;
  }
}

void PrintBytes(std::ostream& os, size_t bytes) {
  if (bytes < 1024) {
    os << bytes << " B";
  } else {
    os << bytes / 1024.0 << " KiB";
  }
}

void PrintShapeDims(std::ostream& os, const std::vector<size_t>& dims, boost::optional<BankDimension> bank_dim) {
  os << "(";
  for (size_t i = 0; i < dims.size(); i++) {
    if (i > 0) {
      os << ", ";
    }
    if (bank_dim && bank_dim->dim_pos == i) {
      os << "{" << dims[i] << "}";
    } else {
      os << dims[i];
    }
  }
  os << ")";
}

struct DefaultCodec : Codec {
  explicit DefaultCodec(const TensorShape* shape) : Codec(shape) {}
  int64_t byte_size() const final { return shape_->sizes_product_bytes(); }
  boost::optional<size_t> sparse_dim() const final { return boost::none; }
};

class CodecRegistry {
 public:
  static CodecRegistry* Instance() {
    static CodecRegistry registry;
    return &registry;
  }

  void Register(const std::string& name, const Codec::Factory& factory) {  //
    registry_[name] = factory;
  }

  std::unique_ptr<Codec> Resolve(const TensorShape& shape) {  //
    return registry_.at(shape.codec)(&shape);
  }

 private:
  CodecRegistry() {
    registry_[""] = [](auto shape) { return std::make_unique<stripe::DefaultCodec>(shape); };
  }

 private:
  std::unordered_map<std::string, Codec::Factory> registry_;
};

}  // namespace

Taggable::Taggable() : impl_(new Impl()) {}

Taggable::~Taggable() = default;

Taggable::Taggable(const Taggable& rhs) { set_attrs(rhs); }

Taggable& Taggable::operator=(const Taggable& rhs) {
  set_attrs(rhs);
  return *this;
}

void Taggable::set_tag(const std::string& tag) { impl_->attrs.emplace(tag, Void{}); }

void Taggable::add_tags(const Tags& to_add) {
  for (const auto& tag : to_add) {
    impl_->attrs.emplace(tag, Void{});
  }
}

void Taggable::clear_tags() { impl_->attrs.clear(); }

void Taggable::remove_tag(const std::string& tag) { impl_->attrs.erase(tag); }

void Taggable::remove_tags(const Tags& to_remove) {
  for (const auto& tag : to_remove) {
    impl_->attrs.erase(tag);
  }
}

void Taggable::set_tags(const Tags& tags) {
  impl_->attrs.clear();
  add_tags(tags);
}

bool Taggable::has_tag(const std::string& tag) const { return impl_->attrs.count(tag); }

bool Taggable::has_tags(const Tags& to_find) const {
  for (const auto& tag : to_find) {
    if (impl_->attrs.count(tag) == 0) {
      return false;
    }
  }
  return true;
}

bool Taggable::has_any_tags(const Tags& to_find) const {
  for (const auto& tag : to_find) {
    if (impl_->attrs.count(tag) == 1) {
      return true;
    }
  }
  return false;
}

class TagVisitorVisitor : public boost::static_visitor<> {
 public:
  std::string name;
  TagVisitor* inner;
  void operator()(const Void& v) const { inner->Visit(name); }
  void operator()(const bool& v) const { inner->Visit(name, v); }
  void operator()(const int64_t& v) const { inner->Visit(name, v); }
  void operator()(const double& v) const { inner->Visit(name, v); }
  void operator()(const std::string& v) const { inner->Visit(name, v); }
  void operator()(const google::protobuf::Any& v) const { inner->Visit(name, v); }
};

bool Taggable::any_tags() const { return !impl_->attrs.empty(); }

void Taggable::visit_tags(TagVisitor* visitor) const {
  TagVisitorVisitor outer;
  outer.inner = visitor;
  for (const auto& kvp : impl_->attrs) {
    outer.name = kvp.first;
    boost::apply_visitor(outer, kvp.second);
  }
}

void Taggable::set_attr(const std::string& name) { impl_->attrs.emplace(name, Void{}); }

void Taggable::set_attr(const std::string& name, bool value) { impl_->attrs.emplace(name, value); }

void Taggable::set_attr(const std::string& name, int64_t value) { impl_->attrs.emplace(name, value); }

void Taggable::set_attr(const std::string& name, double value) { impl_->attrs.emplace(name, value); }

void Taggable::set_attr(const std::string& name, const std::string& value) { impl_->attrs.emplace(name, value); }

void Taggable::set_attr(const std::string& name, const Any& value) { impl_->attrs.emplace(name, value); }

bool Taggable::has_attr(const std::string& name) const { return impl_->attrs.count(name); }

void Taggable::set_attrs(const Taggable& rhs) {
  if (this != &rhs) {
    impl_.reset(new Impl(*rhs.impl_));
  }
}

bool Taggable::get_attr_bool(const std::string& name) const { return boost::get<bool>(impl_->attrs[name]); }

int64_t Taggable::get_attr_int(const std::string& name) const { return boost::get<int64_t>(impl_->attrs[name]); }

double Taggable::get_attr_float(const std::string& name) const { return boost::get<double>(impl_->attrs[name]); }

std::string Taggable::get_attr_str(const std::string& name) const {
  return boost::get<std::string>(impl_->attrs[name]);
}

Any Taggable::get_attr_any(const std::string& name) const { return boost::get<Any>(impl_->attrs[name]); }

bool Taggable::get_attr_bool(const std::string& name, bool def) const {
  return has_attr(name) ? get_attr_bool(name) : def;
}

int64_t Taggable::get_attr_int(const std::string& name, int64_t def) const {
  return has_attr(name) ? get_attr_int(name) : def;
}

double Taggable::get_attr_float(const std::string& name, double def) const {
  return has_attr(name) ? get_attr_float(name) : def;
}

std::string Taggable::get_attr_str(const std::string& name, const std::string& def) const {
  return has_attr(name) ? get_attr_str(name) : def;
}

std::shared_ptr<Load> Load::Downcast(const std::shared_ptr<Statement>& stmt) {  //
  return std::dynamic_pointer_cast<Load>(stmt);
}

std::shared_ptr<Store> Store::Downcast(const std::shared_ptr<Statement>& stmt) {  //
  return std::dynamic_pointer_cast<Store>(stmt);
}

std::shared_ptr<LoadIndex> LoadIndex::Downcast(const std::shared_ptr<Statement>& stmt) {  //
  return std::dynamic_pointer_cast<LoadIndex>(stmt);
}

std::shared_ptr<Intrinsic> Intrinsic::Downcast(const std::shared_ptr<Statement>& stmt) {  //
  return std::dynamic_pointer_cast<Intrinsic>(stmt);
}

std::shared_ptr<Special> Special::Downcast(const std::shared_ptr<Statement>& stmt) {  //
  return std::dynamic_pointer_cast<Special>(stmt);
}

std::shared_ptr<Constant> Constant::Downcast(const std::shared_ptr<Statement>& stmt) {  //
  return std::dynamic_pointer_cast<Constant>(stmt);
}

std::shared_ptr<Block> Block::Downcast(const std::shared_ptr<Statement>& stmt) {  //
  return std::dynamic_pointer_cast<Block>(stmt);
}

std::string to_string(RefDir dir) {
  switch (dir) {
    case RefDir::None:
      return "none";
    case RefDir::In:
      return "in";
    case RefDir::Out:
      return "out";
    case RefDir::InOut:
      return "inout";
    default:
      return "<invalid dir>";
  }
}

std::string to_string(const Device& dev) {
  std::stringstream ss;
  ss << dev;
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, const Device& dev) {
  os << dev.name;
  if (dev.units.size()) {
    os << '[';
    bool unit_sep = false;
    for (const auto& unit : dev.units) {
      if (unit_sep) {
        os << ", ";
      }
      unit_sep = true;
      os << to_string(unit);
    }
    os << ']';
  }
  return os;
}

std::string to_string(const Location& loc) {
  std::stringstream ss;
  ss << loc;
  return ss.str();
}

std::ostream& operator<<(std::ostream& os, const Location& loc) {
  bool dev_sep = false;
  for (const auto& dev : loc.devs) {
    if (dev_sep) {
      os << '/';
    }
    dev_sep = true;
    os << dev;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const Load& op) {
  os << op.into << " = load(" << op.from << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Store& op) {
  os << op.into << " = store(" << op.from << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const LoadIndex& op) {
  os << op.into << " = load_index(" << op.from << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Intrinsic& op) {
  if (op.outputs.size() > 1) {
    os << "(";
  }
  for (size_t i = 0; i < op.outputs.size(); i++) {
    if (i > 0) {
      os << ", ";
    }
    os << op.outputs[i];
  }
  if (op.outputs.size() > 1) {
    os << ")";
  }
  os << " = " << op.name << "(";
  for (size_t i = 0; i < op.inputs.size(); i++) {
    if (i > 0) {
      os << ", ";
    }
    os << op.inputs[i];
  }
  os << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Special& op) {
  if (op.outputs.size() > 1) {
    os << "(";
  }
  for (size_t i = 0; i < op.outputs.size(); i++) {
    if (i > 0) {
      os << ", ";
    }
    os << op.outputs[i];
  }
  if (op.outputs.size() > 1) {
    os << ")";
  }
  os << " = " << op.name << "(";
  for (size_t i = 0; i < op.inputs.size(); i++) {
    if (i > 0) {
      os << ", ";
    }
    os << op.inputs[i];
  }
  os << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Constant& op) {
  os << op.name << " = ";
  switch (op.type) {
    case ConstType::Integer:
      os << "(int)" << op.iconst;
      break;
    case ConstType::Float:
      os << "(float)" << op.fconst;
      break;
    default:
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const Refinement& ref) {
  os << PrintRefinement{ref};
  return os;
}

std::ostream& operator<<(std::ostream& os, const PrintRefinement& printer) {
  const auto& ref = printer.ref;
  auto impl = Accessor::impl(ref);
  if (impl->attrs.size()) {
    for (const auto& attr : impl->attrs) {
      os << "#" << attr.first << " ";
    }
  }
  os << to_string(ref.dir);
  if (ref.interior_shape.is_const) {
    os << " const";
  }
  if (ref.from.empty()) {
    os << " new@0x";
    os << std::hex << std::setw(8) << std::setfill('0') << ref.offset << std::dec;
  }
  os << " " << ref.into();
  if (ref.into() != ref.from) {
    if (!ref.from.empty()) {
      os << " = " << ref.from;
    }
  }
  if (!ref.location.empty()) {
    os << "<" << ref.location << ">";
  }
  os << "[";
  for (size_t i = 0; i < ref.access.size(); i++) {
    const auto& access = ref.access[i];
    if (i > 0) {
      os << ", ";
    }
    os << access.toString();
  }
  os << "]";
  if (!ref.agg_op.empty()) {
    os << ":" << ref.agg_op;
  }
  os << " " << to_string(ref.interior_shape.type);
  if (!ref.interior_shape.layout.empty()) {
    os << "[" << ref.interior_shape.layout << "]";
  }
  if (!ref.interior_shape.codec.empty()) {
    os << "[" << ref.interior_shape.codec << "]";
  }
  os << ":I";
  PrintShapeDims(os, ref.interior_shape.sizes(), ref.bank_dim);
  os << ":";
  PrintShapeDims(os, ref.interior_shape.strides(), ref.bank_dim);
  os << ":";
  PrintBytes(os, ref.interior_shape.sizes_product_bytes());
  if (!ref.interior_shape.codec.empty()) {
    os << "(";
    PrintBytes(os, Codec::Resolve(ref.interior_shape)->byte_size());
    os << ")";
  }
  if (printer.block && !ref.from.empty()) {
    os << ", E";
    auto exterior_shape = printer.block->exterior_shape(ref.into());
    PrintShapeDims(os, exterior_shape.sizes(), ref.bank_dim);
    os << ":";
    PrintBytes(os, exterior_shape.sizes_product_bytes());
    if (!exterior_shape.codec.empty()) {
      os << "(";
      PrintBytes(os, Codec::Resolve(exterior_shape)->byte_size());
      os << ")";
    }
  }
  if (ref.cache_unit) {
    os << ", cache[" << *ref.cache_unit << "]";
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const Block& block) {
  DepsMap deps;
  PrintStmt(os, block, 0, 0, deps);
  return os;
}

std::vector<size_t> Block::sorted_idx_ranges() {
  std::vector<size_t> ranges;
  for (const auto& idx : idxs) {
    ranges.push_back(idx.range);
  }
  std::sort(ranges.begin(), ranges.end());
  return ranges;
}

std::shared_ptr<Block> Block::SubBlock(size_t pos, bool reverse) const {
  if (stmts.size() <= pos) {
    throw std::out_of_range(str(boost::format("SubBlock(%1%) is out of range") % pos));
  }
  auto it = stmts.begin();
  size_t new_pos = reverse ? (stmts.size() - pos - 1) : pos;
  for (size_t i = 0; i < new_pos; i++) {
    ++it;
  }
  return Block::Downcast(*it);
}

std::vector<std::string> Block::buffer_reads() const {
  std::vector<std::string> results;
  for (const auto& ref : refs) {
    if (IsReadDir(ref.dir)) {
      results.push_back(ref.from);
    }
  }
  return results;
}

std::vector<std::string> Block::buffer_writes() const {
  std::vector<std::string> results;
  for (const auto& ref : refs) {
    if (IsWriteDir(ref.dir)) {
      results.push_back(ref.from);
    }
  }
  return results;
}

std::vector<const Refinement*> Block::ref_ins(bool inout) const {
  std::vector<const Refinement*> results;
  for (const auto& ref : refs) {
    if (ref.dir == RefDir::In || (inout && ref.dir == RefDir::InOut)) {
      results.push_back(&ref);
    }
  }
  return results;
}

std::vector<const Refinement*> Block::ref_outs(bool inout) const {
  std::vector<const Refinement*> results;
  for (const auto& ref : refs) {
    if (ref.dir == RefDir::Out || (inout && ref.dir == RefDir::InOut)) {
      results.push_back(&ref);
    }
  }
  return results;
}

std::vector<const Refinement*> Block::ref_inouts() const {
  std::vector<const Refinement*> results;
  for (const auto& ref : refs) {
    if (ref.dir == RefDir::InOut) {
      results.push_back(&ref);
    }
  }
  return results;
}

std::vector<Refinement*> Block::ref_ins(bool inout) {
  std::vector<Refinement*> results;
  for (auto& ref : refs) {
    if (ref.dir == RefDir::In || (inout && ref.dir == RefDir::InOut)) {
      results.push_back(&ref.mut());
    }
  }
  return results;
}

std::vector<Refinement*> Block::ref_outs(bool inout) {
  std::vector<Refinement*> results;
  for (auto& ref : refs) {
    if (ref.dir == RefDir::Out || (inout && ref.dir == RefDir::InOut)) {
      results.push_back(&ref.mut());
    }
  }
  return results;
}

std::vector<Refinement*> Block::ref_inouts() {
  std::vector<Refinement*> results;
  for (auto& ref : refs) {
    if (ref.dir == RefDir::InOut) {
      results.push_back(&ref.mut());
    }
  }
  return results;
}

void Block::erase_stmt(const StatementIt& it) {
  stmts.erase(it);
  // Dependencies become invalid after a stmt is removed
  for (auto& stmt : stmts) {
    stmt->deps.clear();
  }
}

void Block::erase_stmts(const StatementIt& begin, const StatementIt& end) {
  stmts.erase(begin, end);
  // Dependencies become invalid after a stmt is removed
  for (auto& stmt : stmts) {
    stmt->deps.clear();
  }
}

std::ostream& operator<<(std::ostream& os, const Index& idx) {
  auto impl = Accessor::impl(idx);
  if (impl->attrs.size()) {
    os << "(";
    for (const auto& attr : impl->attrs) {
      os << "#" << attr.first << " ";
    }
  }
  os << idx.name;
  if (idx.affine.constant() || !idx.affine.getMap().empty()) {
    os << " = " << idx.affine.toString();
  } else {
    os << ":" << idx.range;
  }
  if (impl->attrs.size()) {
    os << ")";
  }
  return os;
}

bool operator==(const BankDimension& lhs, const BankDimension& rhs) {
  return std::tie(lhs.dim_pos) == std::tie(rhs.dim_pos);
}

bool operator==(const Index& lhs, const Index& rhs) {
  return std::tie(lhs.name, lhs.range, lhs.affine) ==  //
         std::tie(rhs.name, rhs.range, rhs.affine);
}

bool operator==(const Device& lhs, const Device& rhs) {
  return std::tie(lhs.name, lhs.units) == std::tie(rhs.name, rhs.units);
}

bool operator!=(const Device& lhs, const Device& rhs) {
  return std::tie(lhs.name, lhs.units) != std::tie(rhs.name, rhs.units);
}

bool operator<(const Device& lhs, const Device& rhs) {
  return std::tie(lhs.name, lhs.units) < std::tie(rhs.name, rhs.units);
}

Device PartialEval(const Device& dev, const std::map<std::string, std::int64_t>& values) {
  Device result;
  result.name = dev.name;
  for (const auto& unit : dev.units) {
    result.units.emplace_back(unit.partial_eval(values));
  }
  return result;
}

bool operator==(const Location& lhs, const Location& rhs) { return lhs.devs == rhs.devs; }

bool operator!=(const Location& lhs, const Location& rhs) { return lhs.devs != rhs.devs; }

bool operator<(const Location& lhs, const Location& rhs) { return lhs.devs < rhs.devs; }

Location AddDeviceUnits(const Location& loc1, const Location& loc2) {
  Location result;
  for (auto dev1 = loc1.devs.begin(), dev2 = loc2.devs.begin(); dev1 != loc1.devs.end() && dev2 != loc2.devs.end();
       ++dev1, ++dev2) {
    if (dev1 == loc1.devs.end() || dev2 == loc2.devs.end() || dev1->name != dev2->name ||
        dev1->units.size() != dev2->units.size()) {
      throw std::runtime_error{"Incompatible addition of differently-shaped locations: " + to_string(loc1) +
                               " != " + to_string(loc2)};
    }
    auto rd = result.devs.emplace(result.devs.end(), Device{dev1->name});
    rd->units.reserve(dev1->units.size());
    for (auto unit1 = dev1->units.begin(), unit2 = dev2->units.begin(); unit1 != dev1->units.end(); ++unit1, ++unit2) {
      rd->units.emplace_back(*unit1 + *unit2);
    }
  }
  return result;
}

Location AppendLocations(const Location& loc1, const Location& loc2) {
  Location result{loc1};
  result.devs.insert(result.devs.end(), loc2.devs.begin(), loc2.devs.end());
  return result;
}

Location RealizeLocation(const Location& loc, const Block& block) {
  Location result{loc};
  for (auto& dev : result.devs) {
    for (auto& unit : dev.units) {
      std::map<std::string, Affine> tag_map;
      for (const auto& name_coeff : unit.getMap()) {
        if (name_coeff.first.size() && name_coeff.first[0] == '#') {
          auto tag = name_coeff.first.substr(1);
          for (const auto& idx : block.idxs) {
            if (idx.has_tag(tag)) {
              tag_map[name_coeff.first] = idx.name;
              break;
            }
          }
        }
      }
      unit.substitute(tag_map);
    }
  }
  return result;
}

bool operator==(const Location& loc, const std::string& pattern) {
  static const std::regex valid_re{R"(((^|/)(\w+|\*)(\[\s*((\d+|\*)(\s*,\s*(\d+|\*))*)?\s*\])?)*)"};
  static const std::regex devs_re{R"((?:^|/)(\w+|\*)(\[([^\[\]/]*)\])?)"};
  static const std::regex units_re{R"((?:^|,)\s*(\d+|\*)\s*)"};

  // N.B. This is definitely not the fastest pattern parser we could
  // write; the parsing here is simple enough that we could do it with
  // C-string logic.
  //
  // We're implementing the pattern parser this way because it's
  // faster to code and much more obviously-correct, and we don't
  // really need low-level performance here.

  if (!std::regex_match(pattern, valid_re)) {
    throw std::runtime_error{"Invalid location pattern: " + pattern};
  }

  std::vector<std::pair<std::string, boost::optional<std::vector<boost::optional<std::int64_t>>>>> devs;

  auto devs_begin = std::sregex_iterator{pattern.begin(), pattern.end(), devs_re};
  auto re_end = std::sregex_iterator{};

  for (auto dit = devs_begin; dit != re_end; ++dit) {
    std::smatch dev_match = *dit;
    auto dev_units_it = devs.emplace(devs.end(), dev_match[1], boost::none);
    if (dev_match[2].first != dev_match[2].second) {
      dev_units_it->second.emplace();
      auto units_begin = std::sregex_iterator{dev_match[3].first, dev_match[3].second, units_re};
      for (auto uit = units_begin; uit != re_end; ++uit) {
        std::smatch unit_match = *uit;
        if (unit_match[1] == "*") {
          dev_units_it->second->emplace_back(boost::none);
        } else {
          dev_units_it->second->emplace_back(std::stoi(unit_match[1]));
        }
      }
    }
  }

  // Compare against the pattern.
  if (loc.devs.size() != devs.size()) {
    return false;
  }

  auto ldit = loc.devs.begin();
  auto pdit = devs.begin();
  for (; ldit != loc.devs.end(); ++ldit, ++pdit) {
    if (pdit->first != "*" && pdit->first != ldit->name) {
      return false;
    }
    if (!pdit->second) {
      continue;
    }
    if (pdit->second->size() != ldit->units.size()) {
      return false;
    }
    auto luit = ldit->units.begin();
    auto puit = pdit->second->begin();
    for (; luit != ldit->units.end(); ++puit, ++luit) {
      if (!*puit) {
        continue;
      }
      if (!luit->isConstant()) {
        return false;
      }
      if (luit->constant() != **puit) {
        return false;
      }
    }
  }
  return true;
}

Location PartialEval(const Location& loc, const std::map<std::string, std::int64_t>& values) {
  Location result;
  for (const auto& dev : loc.devs) {
    result.devs.emplace_back(PartialEval(dev, values));
  }
  return result;
}

class CloneVisitor : RewriteStmtVisitor {
 public:
  explicit CloneVisitor(int depth) : depth_(depth) {}
  Load* Visit(const Load& x) { return new Load(x); }
  Store* Visit(const Store& x) { return new Store(x); }
  LoadIndex* Visit(const LoadIndex& x) { return new LoadIndex(x); }
  Constant* Visit(const Constant& x) { return new Constant(x); }
  Special* Visit(const Special& x) { return new Special(x); }
  Intrinsic* Visit(const Intrinsic& x) { return new Intrinsic(x); }
  Block* Visit(const Block& x) {
    auto ret = new Block(x);
    if (depth_ == 0) {
      return ret;
    }
    depth_--;
    std::unordered_map<Statement*, StatementIt> dep_map;  // src-block ptr -> clone-block StatementIt
    for (StatementIt sit = ret->stmts.begin(); sit != ret->stmts.end(); ++sit) {
      Statement* clone = (*sit)->Accept(this);
      for (auto& dit : clone->deps) {
        dit = dep_map.at(dit->get());
      }
      dep_map[sit->get()] = sit;
      sit->reset(clone);
    }
    depth_++;
    return ret;
  }

 private:
  int depth_;
};

std::shared_ptr<Block> CloneBlock(const Block& orig, int depth) {
  CloneVisitor visitor(depth);
  return std::shared_ptr<Block>(visitor.Visit(orig));
}

const Index* Block::idx_by_name(const std::string& name) const {
  auto it = std::find_if(idxs.begin(), idxs.end(), [&name](const Index& idx) { return idx.name == name; });
  if (it == idxs.end()) {
    return nullptr;
  }
  return &*it;
}

Index* Block::idx_by_name(const std::string& name) {
  auto it = std::find_if(idxs.begin(), idxs.end(), [&name](const Index& idx) { return idx.name == name; });
  if (it == idxs.end()) {
    return nullptr;
  }
  return &*it;
}

size_t Block::idxs_product() const {
  size_t product = 1;
  for (const auto& idx : idxs) {
    if (idx.affine == Affine()) {
      product *= idx.range;
    }
  }
  return product;
}

std::set<const Index*> Block::accumulation_idxs(bool inout) const {
  std::set<const Index*> ret;
  for (const auto& idx : idxs) {
    bool used = false;
    for (const auto& ref : ref_outs(inout)) {
      for (const auto& access : ref->access) {
        if (access.getMap().count(idx.name)) {
          used = true;
        }
      }
    }
    if (!used) {
      ret.insert(&idx);
    }
  }
  return ret;
}

std::set<Refinement>::iterator Block::ref_by_into(const std::string& ref_name, bool fail) {
  auto it = refs.find(ref_name);
  if (fail && it == refs.end()) {
    throw_with_trace(
        std::runtime_error(str(boost::format("Refinement not found on block '%s' via into: %s") % name % ref_name)));
  }
  return it;
}

std::set<Refinement>::const_iterator Block::ref_by_into(const std::string& ref_name, bool fail) const {
  auto it = refs.find(ref_name);
  if (fail && it == refs.end()) {
    throw_with_trace(
        std::runtime_error(str(boost::format("Refinement not found on block '%s' via into: %s") % name % ref_name)));
  }
  return it;
}

std::set<Refinement>::iterator Block::ref_by_from(const std::string& ref_name, bool fail) {
  auto it = std::find_if(refs.begin(), refs.end(), [&ref_name](const Refinement& ref) { return ref.from == ref_name; });
  if (fail && it == refs.end()) {
    throw_with_trace(
        std::runtime_error(str(boost::format("Refinement not found on block '%s' via from: %s") % name % ref_name)));
  }
  return it;
}

std::set<Refinement>::const_iterator Block::ref_by_from(const std::string& ref_name, bool fail) const {
  auto it = std::find_if(refs.begin(), refs.end(), [&ref_name](const Refinement& ref) { return ref.from == ref_name; });
  if (fail && it == refs.end()) {
    throw_with_trace(
        std::runtime_error(str(boost::format("Refinement not found on block '%s' via from: %s") % name % ref_name)));
  }
  return it;
}

std::set<Refinement>::iterator Block::ref_by_tag(const std::string& tag_name, bool fail) {
  auto it =
      std::find_if(refs.begin(), refs.end(), [&tag_name](const Refinement& ref) { return ref.has_tag(tag_name); });
  if (fail && it == refs.end()) {
    throw_with_trace(
        std::runtime_error(str(boost::format("Refinement not found on block '%s' via tag: %s") % name % tag_name)));
  }
  return it;
}

std::set<Refinement>::const_iterator Block::ref_by_tag(const std::string& tag_name, bool fail) const {
  auto it =
      std::find_if(refs.begin(), refs.end(), [&tag_name](const Refinement& ref) { return ref.has_tag(tag_name); });
  if (fail && it == refs.end()) {
    throw_with_trace(
        std::runtime_error(str(boost::format("Refinement not found on block '%s' via tag: %s") % name % tag_name)));
  }
  return it;
}

std::string Block::unique_ref_name(const std::string& into) const {
  if (ref_by_into(into, false) == refs.end()) {
    return into;
  }
  size_t i = 0;
  for (;;) {
    auto name = str(boost::format("%s_%02zu") % into % i++);
    if (ref_by_into(name, false) == refs.end()) {
      return name;
    }
  }
}

std::string Block::unique_idx_name(const std::string& name) const {
  if (!idx_by_name(name)) {
    return name;
  }
  size_t i = 0;
  for (;;) {
    auto new_name = str(boost::format("%s_%zu") % name % i++);
    if (!idx_by_name(new_name)) {
      return new_name;
    }
  }
}

TensorShape Block::exterior_shape(const std::string& into) const {
  auto it = ref_by_into(into);
  std::map<std::string, size_t> idx_ranges;
  for (const auto& idx : idxs) {
    idx_ranges.emplace(idx.name, idx.range);
  }
  return it->ApplyTile(idx_ranges);
}

Affine Refinement::FlatAccess() const {
  assert(access.size() == interior_shape.dims.size());
  Affine ret;
  for (size_t i = 0; i < access.size(); i++) {
    ret += interior_shape.dims[i].stride * access[i];
  }
  return ret;
}

TensorShape Refinement::ApplyTile(const std::map<std::string, size_t>& tile_by_name) const {
  TensorShape shape = interior_shape;
  for (size_t i = 0; i < access.size(); i++) {
    const auto& aff = access[i];
    int64_t neg = 0;
    int64_t pos = 0;
    for (const auto& kvp : aff.getMap()) {
      if (kvp.first.empty()) {
        continue;
      }
      if (kvp.second > 0) {
        pos += kvp.second * (tile_by_name.at(kvp.first) - 1);
      } else {
        neg += kvp.second * (tile_by_name.at(kvp.first) - 1);
      }
    }
    auto& dim = shape.dims[i];
    dim.size = (dim.size - 1) + pos - neg + 1;
  }
  return shape;
}

const Block* FindBlockByTag(const Block& block, const std::string& tag) {
  if (block.has_tag(tag)) {
    return &block;
  }
  for (const auto& stmt : block.stmts) {
    auto inner = Block::Downcast(stmt);
    if (inner) {
      const Block* out = FindBlockByTag(*inner, tag);
      if (out) {
        return out;
      }
    }
  }
  return nullptr;
}

void FindBlocksByTag(std::vector<const Block*>* into, const Block& block, const std::string& tag) {
  if (block.has_tag(tag)) {
    into->push_back(&block);
  }
  for (const auto& stmt : block.stmts) {
    auto inner = Block::Downcast(stmt);
    if (inner) {
      FindBlocksByTag(into, *inner, tag);
    }
  }
}

const Index* FindIndexByTag(const Block& block, const std::string& tag) {
  for (const auto& idx : block.idxs) {
    if (idx.has_tag(tag)) {
      return &idx;
    }
  }
  return nullptr;
}

bool InsertAfterBlock(Block* parent, Block* sub, std::shared_ptr<Statement> stmt) {
  for (auto it = parent->stmts.begin(); it != parent->stmts.end(); ++it) {
    auto block = Block::Downcast(*it);
    if (block && block.get() == sub) {
      ++it;
      parent->stmts.insert(it, stmt);
      return true;
    }
  }
  return false;
}

Codec::Codec(const TensorShape* shape) : shape_(shape) {  //
}

void Codec::Register(const std::string& name, const Codec::Factory& factory) {  //
  CodecRegistry::Instance()->Register(name, factory);
}

std::unique_ptr<Codec> Codec::Resolve(const TensorShape& shape) {  //
  return CodecRegistry::Instance()->Resolve(shape);
}

bool FromProtoText(const std::string& pbtxt, proto::Program* into) {
  return google::protobuf::TextFormat::ParseFromString(pbtxt, into);
}

}  // namespace stripe
}  // namespace tile
}  // namespace vertexai
