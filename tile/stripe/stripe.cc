// Copyright 2018, Intel Corporation

#include "tile/stripe/stripe.h"

#include <sstream>

#include <boost/format.hpp>

#include "base/util/stream_container.h"
#include "base/util/throw.h"
#include "tile/base/shape.h"

namespace vertexai {
namespace tile {
namespace stripe {

const char* Special::ZERO = "zero";
const char* Special::COPY = "copy";

const char* Intrinsic::ASSIGN = "assign";
const char* Intrinsic::SUM = "add";
const char* Intrinsic::MIN = "min";
const char* Intrinsic::MAX = "max";
const char* Intrinsic::PROD = "mul";

const char* Intrinsic::MUL = "mul";
const char* Intrinsic::ADD = "add";
const char* Intrinsic::EQ = "cmp_eq";
const char* Intrinsic::COND = "cond";

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
                  const Statement* stmt,  //
                  size_t idx,             //
                  const DepsMap& deps) {
  PrintTab(os, depth);
  os << idx;
  if (stmt->deps.size()) {
    os << "[";
    bool first = true;
    for (const auto& it : stmt->deps) {
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
  if (stmt->tags.size()) {
    for (const auto& tag : stmt->tags) {
      os << "#" << tag << " ";
    }
    os << std::endl;
    PrintTab(os, depth);
  }
}

void PrintRefinements(std::ostream& os, const Block& block, size_t depth) {
  if (block.refs.size() > 2) {
    std::map<std::string, const Refinement*> sorted;
    for (const auto& ref : block.refs) {
      sorted.emplace(ref.into, &ref);
    }
    for (const auto& kvp : sorted) {
      PrintTab(os, depth + 2);
      os << *kvp.second << std::endl;
    }
  } else {
    for (const auto& ref : block.refs) {
      PrintTab(os, depth + 2);
      os << ref << std::endl;
    }
  }
}

void PrintBlock(std::ostream& os,    //
                const Block& block,  //
                size_t depth,        //
                size_t block_idx,    //
                const DepsMap& block_deps) {
  os << "block";
  if (!block.location.name.empty()) {
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
  PrintPreStmt(os, depth, &stmt, idx, deps);
  switch (stmt.kind()) {
    case StmtKind::Load:
      os << *dynamic_cast<const Load*>(&stmt) << std::endl;
      break;
    case StmtKind::Store:
      os << *dynamic_cast<const Store*>(&stmt) << std::endl;
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

}  // namespace

std::shared_ptr<Load> Load::Downcast(const std::shared_ptr<Statement>& stmt) {  //
  return std::dynamic_pointer_cast<Load>(stmt);
}

std::shared_ptr<Store> Store::Downcast(const std::shared_ptr<Statement>& stmt) {  //
  return std::dynamic_pointer_cast<Store>(stmt);
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

std::string to_string(const Location& loc) {  //
  return str(boost::format("%s[%s]") % loc.name % loc.unit.toString());
}

std::ostream& operator<<(std::ostream& os, const Location& loc) {
  os << to_string(loc);
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
  if (ref.tags.size()) {
    for (const auto& tag : ref.tags) {
      os << "#" << tag << " ";
    }
  }
  switch (ref.dir) {
    case RefDir::None:
      os << "none";
      break;
    case RefDir::In:
      os << "in";
      break;
    case RefDir::Out:
      os << "out";
      break;
    case RefDir::InOut:
      os << "inout";
      break;
  }
  if (ref.is_const) {
    os << " const";
  }
  if (ref.from.empty()) {
    os << " new@0x";
    os << std::hex << std::setw(8) << std::setfill('0') << ref.offset << std::dec;
  }
  os << " " << ref.into;
  if (ref.into != ref.from) {
    if (!ref.from.empty()) {
      os << " = " << ref.from;
    }
  }
  if (!ref.location.name.empty()) {
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
  os << " " << to_string(ref.interior_shape.type) << "(";
  for (size_t i = 0; i < ref.interior_shape.dims.size(); i++) {
    if (i > 0) {
      os << ", ";
    }
    if (ref.bank_dim && ref.bank_dim->dim_pos == i) {
      os << "{" << ref.interior_shape.dims[i].size << "}";
    } else {
      os << ref.interior_shape.dims[i].size;
    }
  }
  os << "):(";
  for (size_t i = 0; i < ref.interior_shape.dims.size(); i++) {
    if (i > 0) {
      os << ", ";
    }
    if (ref.bank_dim && ref.bank_dim->dim_pos == i) {
      os << "{" << ref.interior_shape.dims[i].stride << "}";
    } else {
      os << ref.interior_shape.dims[i].stride;
    }
  }
  os << "):";
  if (ref.interior_shape.byte_size() < 1024) {
    os << ref.interior_shape.byte_size() << " B";
  } else {
    os << ref.interior_shape.byte_size() / 1024.0 << " KiB";
  }
  if (ref.bank_dim) {
    os << ", banking " << ref.bank_dim->orig_name << " " << ref.bank_dim->orig_shape
       << " on didx=" << ref.bank_dim->dim_pos;
  }
  if (!ref.from.empty() && ref.into != ref.from) {
    os << " // alias";
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const Block& block) {
  DepsMap deps;
  PrintStmt(os, block, 0, 0, deps);
  return os;
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

std::vector<const Refinement*> Block::ref_ins() const {
  std::vector<const Refinement*> results;
  for (const auto& ref : refs) {
    if (ref.dir == RefDir::In) {
      results.push_back(&ref);
    }
  }
  return results;
}

std::vector<const Refinement*> Block::ref_outs() const {
  std::vector<const Refinement*> results;
  for (const auto& ref : refs) {
    if (ref.dir == RefDir::Out) {
      results.push_back(&ref);
    }
  }
  return results;
}

std::ostream& operator<<(std::ostream& os, const Index& idx) {
  if (!idx.tags.empty()) {
    os << "(";
    for (const auto& tag : idx.tags) {
      os << "#" << tag << " ";
    }
  }
  os << idx.name;
  if (idx.affine.constant() || !idx.affine.getMap().empty()) {
    os << " = " << idx.affine.toString();
  } else {
    os << ":" << idx.range;
  }
  if (!idx.tags.empty()) {
    os << ")";
  }
  return os;
}

bool operator==(const BankDimension& lhs, const BankDimension& rhs) {
  return std::tie(lhs.dim_pos, lhs.orig_shape) ==  //
         std::tie(rhs.dim_pos, rhs.orig_shape);
}

bool operator==(const Index& lhs, const Index& rhs) {
  return std::tie(lhs.name, lhs.range, lhs.affine) ==  //
         std::tie(rhs.name, rhs.range, rhs.affine);
}

bool operator==(const Location& lhs, const Location& rhs) {
  return std::tie(lhs.name, lhs.unit) == std::tie(rhs.name, rhs.unit);
}

bool operator!=(const Location& lhs, const Location& rhs) {
  return std::tie(lhs.name, lhs.unit) != std::tie(rhs.name, rhs.unit);
}

bool operator<(const Location& lhs, const Location& rhs) {
  return std::tie(lhs.name, lhs.unit) < std::tie(rhs.name, rhs.unit);
}

class CloneVisitor : RewriteStmtVisitor {
 public:
  explicit CloneVisitor(int depth) : depth_(depth) {}
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
    for (auto& stmt : ret->stmts) {
      stmt.reset(stmt->Accept(this));
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

Affine FromProto(const proto::Affine& affine) {
  Affine ret = affine.offset();
  for (const auto& kvp : affine.terms()) {
    ret += Affine(kvp.first, kvp.second);
  }
  return ret;
}

Location FromProto(const proto::Location& loc) {  //
  return Location{loc.name(), FromProto(loc.unit())};
}

RefDir FromProto(const proto::Refinement::Dir& dir) {
  switch (dir) {
    case proto::Refinement::None:
      return RefDir::None;
    case proto::Refinement::In:
      return RefDir::In;
    case proto::Refinement::Out:
      return RefDir::Out;
    case proto::Refinement::InOut:
      return RefDir::InOut;
    default:
      throw std::runtime_error("Invalid RefDir");
  }
}

std::shared_ptr<Block> FromProto(const proto::Block& block) {
  auto ret = std::make_shared<Block>();
  ret->name = block.name();
  ret->comments = block.comments();
  ret->location = FromProto(block.location());
  for (const auto& pb_idx : block.idxs()) {
    ret->idxs.emplace_back(Index{pb_idx.name(), pb_idx.range(), FromProto(pb_idx.affine())});
  }
  for (const auto& pb_con : block.constraints()) {
    ret->constraints.emplace_back(FromProto(pb_con));
  }
  for (const auto& pb_ref : block.refs()) {
    Refinement ref;
    ref.dir = FromProto(pb_ref.dir());
    ref.from = pb_ref.from();
    ref.into = pb_ref.into();
    for (const auto& pb_off : pb_ref.access()) {
      ref.access.emplace_back(FromProto(pb_off));
    }
    ref.interior_shape = tile::FromProto(pb_ref.shape());
    ref.agg_op = pb_ref.agg_op();
    ref.location = FromProto(pb_ref.location());
    ref.is_const = pb_ref.is_const();
    ref.offset = pb_ref.offset();
    // if (pb_ref.has_bank_dim()) {
    //   ref.bank_dim = pb_ref.bank_dim().value();
    // }
    ret->refs.emplace_back(ref);
  }
  std::vector<StatementIt> stmts;
  stmts.reserve(block.stmts_size());
  for (const auto& pb_stmt : block.stmts()) {
    switch (pb_stmt.op_case()) {
      case proto::Statement::kLoad: {
        auto stmt = std::make_shared<Load>(pb_stmt.load().from(), pb_stmt.load().into());
        stmts.push_back(ret->stmts.emplace(ret->stmts.end(), std::move(stmt)));
      } break;
      case proto::Statement::kStore: {
        auto stmt = std::make_shared<Store>(pb_stmt.store().from(), pb_stmt.store().into());
        stmts.push_back(ret->stmts.emplace(ret->stmts.end(), std::move(stmt)));
      } break;
      case proto::Statement::kConstant:
        switch (pb_stmt.constant().value_case()) {
          case proto::Constant::kIconst: {
            auto stmt = std::make_shared<Constant>(pb_stmt.constant().name(), pb_stmt.constant().iconst());
            stmts.push_back(ret->stmts.emplace(ret->stmts.end(), std::move(stmt)));
          } break;
          case proto::Constant::kFconst: {
            auto stmt = std::make_shared<Constant>(pb_stmt.constant().name(), pb_stmt.constant().fconst());
            stmts.push_back(ret->stmts.emplace(ret->stmts.end(), std::move(stmt)));
          } break;
          default:
            break;
        }
        break;
      case proto::Statement::kSpecial: {
        auto stmt = std::make_shared<Special>();
        stmt->name = pb_stmt.special().name();
        for (const auto& item : pb_stmt.special().params()) {
          stmt->params.push_back(item);
        }
        for (const auto& item : pb_stmt.special().inputs()) {
          stmt->inputs.push_back(item);
        }
        for (const auto& item : pb_stmt.special().outputs()) {
          stmt->outputs.push_back(item);
        }
        stmts.push_back(ret->stmts.emplace(ret->stmts.end(), std::move(stmt)));
      } break;
      case proto::Statement::kIntrinsic: {
        auto stmt = std::make_shared<Intrinsic>();
        stmt->name = pb_stmt.intrinsic().name();
        stmt->type = tile::FromProto(pb_stmt.intrinsic().type());
        for (const auto& item : pb_stmt.intrinsic().inputs()) {
          stmt->inputs.push_back(item);
        }
        for (const auto& item : pb_stmt.intrinsic().outputs()) {
          stmt->outputs.push_back(item);
        }
        stmts.push_back(ret->stmts.emplace(ret->stmts.end(), std::move(stmt)));
      } break;
      case proto::Statement::kBlock: {
        auto stmt = FromProto(pb_stmt.block());
        stmts.push_back(ret->stmts.emplace(ret->stmts.end(), std::move(stmt)));
      } break;
      default:
        break;
    }
    std::shared_ptr<Statement> stmt = *stmts.back();
    for (std::size_t dep_idx : pb_stmt.deps()) {
      stmt->deps.push_back(stmts[dep_idx]);
    }
    for (const auto& tag : pb_stmt.tags()) {
      stmt->tags.emplace(tag);
    }
  }
  return ret;
}

proto::Affine IntoProto(const Affine& affine) {
  proto::Affine ret;
  for (const auto& kvp : affine.getMap()) {
    if (kvp.first.empty()) {
      ret.set_offset(kvp.second);
    } else {
      std::string str = kvp.first;
      (*ret.mutable_terms())[str] = kvp.second;
    }
  }
  return ret;
}

proto::Location IntoProto(const Location& loc) {
  proto::Location ret;
  ret.set_name(loc.name);
  *ret.mutable_unit() = IntoProto(loc.unit);
  return ret;
}

proto::Block IntoProto(const Block& block) {
  proto::Block ret;
  ret.set_name(block.name);
  ret.set_comments(block.comments);
  *ret.mutable_location() = IntoProto(block.location);
  for (const auto& idx : block.idxs) {
    auto pb_idx = ret.add_idxs();
    pb_idx->set_name(idx.name);
    pb_idx->set_range(idx.range);
    *pb_idx->mutable_affine() = IntoProto(idx.affine);
  }
  for (const auto& con : block.constraints) {
    *ret.add_constraints() = IntoProto(con);
  }
  for (const auto& ref : block.refs) {
    auto pb_ref = ret.add_refs();
    switch (ref.dir) {
      case RefDir::None:
        pb_ref->set_dir(proto::Refinement::None);
        break;
      case RefDir::In:
        pb_ref->set_dir(proto::Refinement::In);
        break;
      case RefDir::Out:
        pb_ref->set_dir(proto::Refinement::Out);
        break;
      case RefDir::InOut:
        pb_ref->set_dir(proto::Refinement::InOut);
        break;
    }
    pb_ref->set_from(ref.from);
    pb_ref->set_into(ref.into);
    for (const auto& access : ref.access) {
      *pb_ref->add_access() = IntoProto(access);
    }
    *pb_ref->mutable_shape() = IntoProto(ref.interior_shape);
    pb_ref->set_agg_op(ref.agg_op);
    *pb_ref->mutable_location() = IntoProto(ref.location);
    pb_ref->set_is_const(ref.is_const);
    pb_ref->set_offset(ref.offset);
    // if (ref.bank_dim) {
    //   pb_ref->mutable_bank_dim()->set_value(*ref.bank_dim);
    // }
  }
  std::unordered_map<Statement*, std::size_t> dep_idxs;
  std::size_t stmt_idx = 0;
  for (const auto& stmt : block.stmts) {
    dep_idxs[stmt.get()] = stmt_idx++;
    auto pb_stmt = ret.add_stmts();
    std::vector<std::size_t> deps;
    for (StatementIt dep : stmt->deps) {
      deps.push_back(dep_idxs[dep->get()]);
    }
    std::sort(deps.begin(), deps.end());  // Provide stable output ordering
    for (std::size_t dep : deps) {
      pb_stmt->add_deps(dep);
    }
    for (const auto& tag : stmt->tags) {
      pb_stmt->add_tags(tag);
    }
    switch (stmt->kind()) {
      case StmtKind::Load: {
        auto load = Load::Downcast(stmt);
        auto pb_load = pb_stmt->mutable_load();
        pb_load->set_from(load->from);
        pb_load->set_into(load->into);
      } break;
      case StmtKind::Store: {
        auto store = Store::Downcast(stmt);
        auto pb_store = pb_stmt->mutable_store();
        pb_store->set_from(store->from);
        pb_store->set_into(store->into);
      } break;
      case StmtKind::Constant: {
        auto constant = Constant::Downcast(stmt);
        auto pb_const = pb_stmt->mutable_constant();
        pb_const->set_name(constant->name);
        switch (constant->type) {
          case ConstType::Integer:
            pb_const->set_iconst(constant->iconst);
            break;
          case ConstType::Float:
            pb_const->set_fconst(constant->fconst);
            break;
        }
      } break;
      case StmtKind::Special: {
        auto special = Special::Downcast(stmt);
        auto pb_special = pb_stmt->mutable_special();
        pb_special->set_name(special->name);
        for (const auto& param : special->params) {
          pb_special->add_params(param);
        }
        for (const auto& input : special->inputs) {
          pb_special->add_inputs(input);
        }
        for (const auto& output : special->outputs) {
          pb_special->add_outputs(output);
        }
      } break;
      case StmtKind::Intrinsic: {
        auto intrinsic = Intrinsic::Downcast(stmt);
        auto pb_intrinsic = pb_stmt->mutable_intrinsic();
        pb_intrinsic->set_name(intrinsic->name);
        pb_intrinsic->set_type(tile::IntoProto(intrinsic->type));
        for (const auto& input : intrinsic->inputs) {
          pb_intrinsic->add_inputs(input);
        }
        for (const auto& output : intrinsic->outputs) {
          pb_intrinsic->add_outputs(output);
        }
      } break;
      case StmtKind::Block: {
        auto inner = Block::Downcast(stmt);
        *pb_stmt->mutable_block() = IntoProto(*inner);
      } break;
    }
  }
  return ret;
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
    product *= idx.range;
  }
  return product;
}

std::set<const Index*> Block::accumulation_idxs() const {
  std::set<const Index*> ret;
  for (const auto& idx : idxs) {
    bool used = false;
    for (const auto& ref : ref_outs()) {
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

std::vector<Refinement>::iterator Block::ref_by_into(const std::string& name, bool fail) {
  auto it = std::find_if(refs.begin(), refs.end(), [&name](const Refinement& ref) { return ref.into == name; });
  if (fail && it == refs.end()) {
    throw_with_trace(
        std::runtime_error(str(boost::format("Refinement not found on block '%s' via into: %s") % this->name % name)));
  }
  return it;
}

std::vector<Refinement>::const_iterator Block::ref_by_into(const std::string& name, bool fail) const {
  auto it = std::find_if(refs.begin(), refs.end(), [&name](const Refinement& ref) { return ref.into == name; });
  if (fail && it == refs.end()) {
    throw_with_trace(
        std::runtime_error(str(boost::format("Refinement not found on block '%s' via into: %s") % this->name % name)));
  }
  return it;
}

std::vector<Refinement>::iterator Block::ref_by_from(const std::string& name, bool fail) {
  auto it = std::find_if(refs.begin(), refs.end(), [&name](const Refinement& ref) { return ref.from == name; });
  if (fail && it == refs.end()) {
    throw_with_trace(
        std::runtime_error(str(boost::format("Refinement not found on block '%s' via from: %s") % this->name % name)));
  }
  return it;
}

std::vector<Refinement>::const_iterator Block::ref_by_from(const std::string& name, bool fail) const {
  auto it = std::find_if(refs.begin(), refs.end(), [&name](const Refinement& ref) { return ref.from == name; });
  if (fail && it == refs.end()) {
    throw_with_trace(
        std::runtime_error(str(boost::format("Refinement not found on block '%s' via from: %s") % this->name % name)));
  }
  return it;
}

std::string Block::unique_ref_name(const std::string& into) const {
  if (ref_by_into(into, false) == refs.end()) {
    return into;
  }
  size_t i = 0;
  while (true) {
    auto name = str(boost::format("%s_%zu") % into % i++);
    if (ref_by_into(name, false) == refs.end()) {
      return name;
    }
  }
  // Unreachable
  return "";
}

std::string Block::unique_idx_name(const std::string& name) const {
  if (!idx_by_name(name)) {
    return name;
  }
  size_t i = 0;
  while (true) {
    auto new_name = str(boost::format("%s_%zu") % name % i++);
    if (!idx_by_name(new_name)) {
      return new_name;
    }
  }
  // Unreachable
  return "";
}

TensorShape Block::exterior_shape(const std::string& name) const {
  auto it = ref_by_into(name);
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
    pos += (dim.size - 1);
    dim.size = pos - neg + 1;
  }
  return shape;
}

}  // namespace stripe
}  // namespace tile
}  // namespace vertexai
