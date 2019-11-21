// Copyright 2018, Intel Corporation

#include "tile/stripe/stripe.h"

#include "tile/stripe/impl.h"

namespace vertexai {
namespace tile {
namespace stripe {

void SetAttributes(Taggable* into, const google::protobuf::Map<std::string, proto::Attribute>& attrs) {
  for (const auto& kvp : attrs) {
    switch (kvp.second.attr_case()) {
      case proto::Attribute::kBval:
        into->set_attr(kvp.first, kvp.second.bval());
        break;
      case proto::Attribute::kIval:
        into->set_attr(kvp.first, kvp.second.ival());
        break;
      case proto::Attribute::kFval:
        into->set_attr(kvp.first, kvp.second.fval());
        break;
      case proto::Attribute::kSval:
        into->set_attr(kvp.first, kvp.second.sval());
        break;
      case proto::Attribute::kAny:
        into->set_attr(kvp.first, kvp.second.any());
        break;
      default:
        into->set_attr(kvp.first);
        break;
    }
  }
}

Affine FromProto(const proto::Affine& affine) {
  Affine ret = affine.offset();
  for (const auto& kvp : affine.terms()) {
    ret += Affine(kvp.first, kvp.second);
  }
  return ret;
}

Device FromProto(const proto::Device& dev) {
  Device result;
  result.name = dev.name();
  for (const auto& unit : dev.units()) {
    result.units.emplace_back(FromProto(unit));
  }
  return result;
}

std::vector<Device> FromProto(const google::protobuf::RepeatedPtrField<proto::Device>& devs) {
  std::vector<Device> result;
  for (const auto& dev : devs) {
    result.emplace_back(FromProto(dev));
  }
  return result;
}

Location FromProto(const proto::Location& loc) {
  Location result;
  result.devs = FromProto(loc.devs());
  return result;
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

Tags FromProto(const google::protobuf::RepeatedPtrField<std::string>& pb_tags) {
  Tags tags;
  for (const auto& tag : pb_tags) {
    tags.emplace(tag);
  }
  return tags;
}

Index FromProto(const proto::Index& idx) {
  auto name = idx.name();
  auto range = idx.range();
  auto affine = FromProto(idx.affine());
  Index ret{name, range, affine};
  SetAttributes(&ret, idx.attrs());
  return ret;
}

Refinement FromProto(const std::string& into, const proto::Refinement& ref) {
  auto ret = Refinement::FromInto(into);
  ret.dir = FromProto(ref.dir());
  ret.from = ref.from();
  for (const auto& pb_off : ref.access()) {
    ret.access.emplace_back(FromProto(pb_off));
  }
  ret.interior_shape = tile::FromProto(ref.interior_shape());
  ret.agg_op = ref.agg_op();
  ret.location = FromProto(ref.loc());
  ret.offset = ref.offset();
  // if (ref.has_bank_dim()) {
  //   ref.bank_dim = ref.bank_dim().value();
  // }
  SetAttributes(&ret, ref.attrs());
  return ret;
}

std::shared_ptr<Load> FromProto(const proto::Load& op) {
  auto from = op.from();
  auto into = op.into();
  return std::make_shared<Load>(from, into);
}

std::shared_ptr<Store> FromProto(const proto::Store& op) {
  auto from = op.from();
  auto into = op.into();
  return std::make_shared<Store>(from, into);
}

std::shared_ptr<LoadIndex> FromProto(const proto::LoadIndex& op) {
  auto from = FromProto(op.from());
  auto into = op.into();
  return std::make_shared<LoadIndex>(from, into);
}

std::shared_ptr<Intrinsic> FromProto(const proto::Intrinsic& op) {
  auto ret = std::make_shared<Intrinsic>();
  ret->name = op.name();
  ret->type = tile::FromProto(op.type());
  for (const auto& item : op.inputs()) {
    ret->inputs.push_back(item);
  }
  for (const auto& item : op.outputs()) {
    ret->outputs.push_back(item);
  }
  return ret;
}

std::shared_ptr<Special> FromProto(const proto::Special& op) {
  auto ret = std::make_shared<Special>();
  ret->name = op.name();
  for (const auto& item : op.inputs()) {
    ret->inputs.push_back(item);
  }
  for (const auto& item : op.outputs()) {
    ret->outputs.push_back(item);
  }
  for (const auto& item : op.int_params()) {
    ret->int_params.emplace(item);
  }
  for (const auto& item : op.str_params()) {
    ret->str_params.emplace(item);
  }
  return ret;
}

std::shared_ptr<Constant> FromProto(const proto::Constant& op) {
  switch (op.value_case()) {
    case proto::Constant::kIconst:
      return std::make_shared<Constant>(op.name(), op.iconst());
    case proto::Constant::kFconst:
      return std::make_shared<Constant>(op.name(), op.fconst());
    default:
      throw std::runtime_error("Invalid ConstantType");
  }
}

std::shared_ptr<Statement> FromProto(const proto::Statement& stmt) {
  switch (stmt.op_case()) {
    case proto::Statement::kLoad:
      return FromProto(stmt.load());
    case proto::Statement::kStore:
      return FromProto(stmt.store());
    case proto::Statement::kLoadIndex:
      return FromProto(stmt.load_index());
    case proto::Statement::kIntrinsic:
      return FromProto(stmt.intrinsic());
    case proto::Statement::kConstant:
      return FromProto(stmt.constant());
    case proto::Statement::kSpecial:
      return FromProto(stmt.special());
    case proto::Statement::kBlock:
      return FromProto(stmt.block());
    default:
      throw std::runtime_error("Invalid op_type");
  }
}

std::shared_ptr<Block> FromProto(const proto::Block& block) {
  auto ret = std::make_shared<Block>();
  ret->name = block.name();
  ret->comments = block.comments();
  ret->location = FromProto(block.loc());
  for (const auto& idx : block.idxs()) {
    ret->idxs.emplace_back(FromProto(idx));
  }
  for (const auto& con : block.constraints()) {
    ret->constraints.emplace_back(FromProto(con));
  }
  for (const auto& ref : block.refs()) {
    ret->refs.emplace(FromProto(ref.first, ref.second));
  }
  std::vector<StatementIt> stmts;
  stmts.reserve(block.stmts_size());
  for (const auto& pb_stmt : block.stmts()) {
    auto stmt = FromProto(pb_stmt);
    stmts.push_back(ret->stmts.emplace(ret->stmts.end(), stmt));
    for (size_t dep_idx : pb_stmt.deps()) {
      stmt->deps.push_back(stmts[dep_idx]);
    }
    SetAttributes(stmt.get(), pb_stmt.attrs());
  }
  return ret;
}

Buffer FromProto(const proto::Buffer& buffer) {
  Buffer ret;
  for (const auto& section : buffer.sections()) {
    ret.sections.emplace(section.first, section.second);
  }
  return ret;
}

std::shared_ptr<Program> FromProto(const proto::Program& program) {
  auto ret = std::make_shared<Program>();
  ret->entry = Block::Downcast(FromProto(program.entry()));
  for (const auto& buf : program.buffers()) {
    ret->buffers.emplace(buf.first, FromProto(buf.second));
  }
  SetAttributes(ret->entry.get(), program.entry().attrs());
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

proto::Device IntoProto(const Device& dev) {
  proto::Device ret;
  ret.set_name(dev.name);
  for (const auto& unit : dev.units) {
    *ret.add_units() = IntoProto(unit);
  }
  return ret;
}

proto::Location IntoProto(const Location& loc) {
  proto::Location ret;
  for (const auto& dev : loc.devs) {
    *ret.add_devs() = IntoProto(dev);
  }
  return ret;
}

struct AttrValueVisitor {
  proto::Attribute operator()(const Void& x) const {
    proto::Attribute ret;
    return ret;
  }

  proto::Attribute operator()(bool x) const {
    proto::Attribute ret;
    ret.set_bval(x);
    return ret;
  }

  proto::Attribute operator()(int64_t x) const {
    proto::Attribute ret;
    ret.set_ival(x);
    return ret;
  }

  proto::Attribute operator()(double x) const {
    proto::Attribute ret;
    ret.set_fval(x);
    return ret;
  }

  proto::Attribute operator()(const std::string& x) const {
    proto::Attribute ret;
    ret.set_sval(x);
    return ret;
  }

  proto::Attribute operator()(const google::protobuf::Any& x) const {
    proto::Attribute ret;
    *ret.mutable_any() = x;
    return ret;
  }
};

proto::Refinement::Dir IntoProto(const RefDir& dir) {
  switch (dir) {
    case RefDir::None:
      return proto::Refinement::None;
    case RefDir::In:
      return proto::Refinement::In;
    case RefDir::Out:
      return proto::Refinement::Out;
    case RefDir::InOut:
      return proto::Refinement::InOut;
    default:
      throw std::runtime_error("Invalid RefDir");
  }
}

proto::Refinement IntoProto(const Refinement& ref) {
  proto::Refinement ret;
  ret.set_dir(IntoProto(ref.dir));
  ret.set_from(ref.from);
  for (const auto& access : ref.access) {
    *ret.add_access() = IntoProto(access);
  }
  *ret.mutable_interior_shape() = IntoProto(ref.interior_shape);
  ret.set_agg_op(ref.agg_op);
  *ret.mutable_loc() = IntoProto(ref.location);
  ret.set_offset(ref.offset);
  // if (ref.bank_dim) {
  //   ret.mutable_bank_dim()->set_value(*ref.bank_dim);
  // }
  AttrValueVisitor visitor;
  for (const auto& attr : Accessor::impl(ref)->attrs) {
    (*ret.mutable_attrs())[attr.first] = std::visit(visitor, attr.second);
  }
  return ret;
}

proto::Load IntoProto(const Load& op) {
  proto::Load ret;
  ret.set_from(op.from);
  ret.set_into(op.into);
  return ret;
}

proto::Store IntoProto(const Store& op) {
  proto::Store ret;
  ret.set_from(op.from);
  ret.set_into(op.into);
  return ret;
}

proto::LoadIndex IntoProto(const LoadIndex& op) {
  proto::LoadIndex ret;
  *ret.mutable_from() = IntoProto(op.from);
  ret.set_into(op.into);
  return ret;
}

proto::Special IntoProto(const Special& op) {
  proto::Special ret;
  ret.set_name(op.name);
  for (const auto& input : op.inputs) {
    ret.add_inputs(input);
  }
  for (const auto& output : op.outputs) {
    ret.add_outputs(output);
  }
  for (const auto& param : op.int_params) {
    (*ret.mutable_int_params())[param.first] = param.second;
  }
  for (const auto& param : op.str_params) {
    (*ret.mutable_str_params())[param.first] = param.second;
  }
  return ret;
}

proto::Constant IntoProto(const Constant& op) {
  proto::Constant ret;
  ret.set_name(op.name);
  switch (op.type) {
    case ConstType::Integer:
      ret.set_iconst(op.iconst);
      break;
    case ConstType::Float:
      ret.set_fconst(op.fconst);
      break;
  }
  return ret;
}

proto::Intrinsic IntoProto(const Intrinsic& op) {
  proto::Intrinsic ret;
  ret.set_name(op.name);
  ret.set_type(tile::IntoProto(op.type));
  for (const auto& input : op.inputs) {
    ret.add_inputs(input);
  }
  for (const auto& output : op.outputs) {
    ret.add_outputs(output);
  }
  return ret;
}

proto::Statement IntoProto(const std::shared_ptr<Statement>& stmt, const std::vector<uint32_t>& deps) {
  proto::Statement ret;
  for (size_t dep : deps) {
    ret.add_deps(dep);
  }
  AttrValueVisitor visitor;
  for (const auto& attr : Accessor::impl(*stmt)->attrs) {
    (*ret.mutable_attrs())[attr.first] = std::visit(visitor, attr.second);
  }
  switch (stmt->kind()) {
    case StmtKind::Load:
      *ret.mutable_load() = IntoProto(*Load::Downcast(stmt));
      break;
    case StmtKind::Store:
      *ret.mutable_store() = IntoProto(*Store::Downcast(stmt));
      break;
    case StmtKind::LoadIndex:
      *ret.mutable_load_index() = IntoProto(*LoadIndex::Downcast(stmt));
      break;
    case StmtKind::Constant:
      *ret.mutable_constant() = IntoProto(*Constant::Downcast(stmt));
      break;
    case StmtKind::Special:
      *ret.mutable_special() = IntoProto(*Special::Downcast(stmt));
      break;
    case StmtKind::Intrinsic:
      *ret.mutable_intrinsic() = IntoProto(*Intrinsic::Downcast(stmt));
      break;
    case StmtKind::Block:
      *ret.mutable_block() = IntoProto(*Block::Downcast(stmt));
      break;
  }
  return ret;
}

proto::Index IntoProto(const Index& idx) {
  proto::Index ret;
  ret.set_name(idx.name);
  ret.set_range(idx.range);
  *ret.mutable_affine() = IntoProto(idx.affine);
  AttrValueVisitor visitor;
  for (const auto& attr : Accessor::impl(idx)->attrs) {
    (*ret.mutable_attrs())[attr.first] = std::visit(visitor, attr.second);
  }
  return ret;
}

proto::Block IntoProto(const Block& block) {
  proto::Block ret;
  ret.set_name(block.name);
  ret.set_comments(block.comments);
  *ret.mutable_loc() = IntoProto(block.location);
  for (const auto& idx : block.idxs) {
    *ret.add_idxs() = IntoProto(idx);
  }
  for (const auto& con : block.constraints) {
    *ret.add_constraints() = IntoProto(con);
  }
  for (const auto& ref : block.refs) {
    (*ret.mutable_refs())[ref.into()] = IntoProto(ref);
  }
  std::unordered_map<Statement*, size_t> deps_by_stmt;
  size_t sid = 0;
  for (const auto& stmt : block.stmts) {
    deps_by_stmt[stmt.get()] = sid++;
    std::vector<uint32_t> deps;
    for (auto dep : stmt->deps) {
      deps.push_back(deps_by_stmt[dep->get()]);
    }
    std::sort(deps.begin(), deps.end());  // Provide stable output ordering
    *ret.add_stmts() = IntoProto(stmt, deps);
  }
  return ret;
}

proto::Buffer IntoProto(const Buffer& buffer) {
  proto::Buffer ret;
  for (const auto& section : buffer.sections) {
    (*ret.mutable_sections())[section.first] = section.second;
  }
  return ret;
}

proto::Program IntoProto(const Program& program) {
  proto::Program ret;
  std::vector<uint32_t> deps;
  auto entry = ret.mutable_entry();
  *entry = IntoProto(program.entry, deps);
  for (const auto& buf : program.buffers) {
    (*ret.mutable_buffers())[buf.first] = IntoProto(buf.second);
  }
  AttrValueVisitor visitor;
  for (const auto& attr : Accessor::impl(*program.entry)->attrs) {
    (*entry->mutable_attrs())[attr.first] = std::visit(visitor, attr.second);
  }
  return ret;
}

}  // namespace stripe
}  // namespace tile
}  // namespace vertexai
