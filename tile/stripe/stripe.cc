#include "tile/stripe/stripe.h"

#include <sstream>

#include "base/util/printstring.h"
#include "base/util/stream_container.h"
#include "tile/base/shape.h"

namespace vertexai {
namespace tile {
namespace stripe {

const char* Intrinsic::ZERO = "ZERO";
const char* Intrinsic::COPY = "COPY";

const char* Intrinsic::ASSIGN = "ASSIGN";
const char* Intrinsic::SUM = "SUM";
const char* Intrinsic::MIN = "MIN";
const char* Intrinsic::MAX = "MAX";
const char* Intrinsic::PROD = "PROD";

const char* Intrinsic::MUL = "MUL";
const char* Intrinsic::ADD = "ADD";
const char* Intrinsic::EQ = "EQ";
const char* Intrinsic::COND = "COND";

static void PrintBlock(std::ostream& os, const Block& block, size_t depth);

static void PrintTab(std::ostream& os, size_t depth) {  //
  os << std::string(depth * 2, ' ');
}

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

std::shared_ptr<BoolAnnotation> BoolAnnotation::Downcast(const std::shared_ptr<Annotation>& ann) {  //
  return std::dynamic_pointer_cast<BoolAnnotation>(ann);
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
      os << op.iconst;
      break;
    case ConstType::Float:
      os << op.fconst;
      break;
    default:
      break;
  }
  return os;
}

static void PrintStatement(std::ostream& os, const std::shared_ptr<Statement>& stmt, size_t depth) {
  switch (stmt->kind()) {
    case StmtKind::Load:
      PrintTab(os, depth);
      os << *Load::Downcast(stmt) << std::endl;
      break;
    case StmtKind::Store:
      PrintTab(os, depth);
      os << *Store::Downcast(stmt) << std::endl;
      break;
    case StmtKind::Intrinsic:
      PrintTab(os, depth);
      os << *Intrinsic::Downcast(stmt) << std::endl;
      break;
    case StmtKind::Special:
      PrintTab(os, depth);
      os << *Special::Downcast(stmt) << std::endl;
      break;
    case StmtKind::Constant:
      PrintTab(os, depth);
      os << *Constant::Downcast(stmt) << std::endl;
      break;
    case StmtKind::Block:
      PrintBlock(os, *Block::Downcast(stmt), depth);
      break;
    default:
      break;
  }
}

static void PrintBlock(std::ostream& os, const Block& block, size_t depth) {
  PrintTab(os, depth);
  os << "block [";
  for (size_t i = 0; i < block.idxs.size(); i++) {
    if (i > 0) {
      os << ", ";
    }
    os << block.idxs[i].name << ":" << block.idxs[i].range;
    if (block.idxs[i].factor != 0) {
      os << ":" << block.idxs[i].factor;
    }
  }
  os << "] (";
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
  for (size_t i = 0; i < block.refs.size(); i++) {
    PrintTab(os, depth + 2);
    const auto& ref = block.refs[i];
    switch (ref.dir) {
      case RefDir::None:
        os << "none ";
        break;
      case RefDir::In:
        os << "in ";
        break;
      case RefDir::Out:
        os << "out ";
        break;
      case RefDir::InOut:
        os << "inout ";
        break;
    }
    if (ref.from.empty()) {
      os << "new ";
    }
    os << ref.into;
    if (ref.into != ref.from) {
      if (!ref.from.empty()) {
        os << " = " << ref.from;
      }
    }
    bool first = true;
    os << "[";
    for (const auto& acc : ref.access) {
      if (!first) {
        os << ", ";
      }
      os << acc.toString();
      first = false;
    }
    os << "]";
    if (!ref.agg_op.empty()) {
      os << ":" << ref.agg_op;
    }
    os << " " << ref.shape;
    if (!ref.from.empty() && ref.into != ref.from) {
      os << " // alias";
    }
    os << std::endl;
  }
  PrintTab(os, depth);
  os << ") {" << std::endl;
  for (const auto& stmt : block.stmts) {
    PrintStatement(os, stmt, depth + 1);
  }
  PrintTab(os, depth);
  os << "}" << std::endl;
}

std::ostream& operator<<(std::ostream& os, const Block& block) {
  PrintBlock(os, block, 0);
  return os;
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
  os << idx.name << ":" << idx.range << ":" << idx.factor;
  return os;
}

bool operator==(const Index& lhs, const Index& rhs) {
  return std::tie(lhs.name, lhs.range, lhs.factor) ==  //
         std::tie(rhs.name, rhs.range, rhs.factor);
}

Affine FromProto(const proto::Affine& affine) {
  Affine r = affine.offset();
  for (const auto& kvp : affine.terms()) {
    r += Affine(kvp.first, kvp.second);
  }
  return r;
}

Block FromProto(const proto::Block& block) {
  Block ret;
  ret.name = block.name();
  ret.comments = block.comments();
  for (const auto& pb_idx : block.idxs()) {
    ret.idxs.emplace_back(Index{
        pb_idx.name(),   //
        pb_idx.range(),  //
        pb_idx.factor()  //
    });
  }
  for (const auto& pb_con : block.constraints()) {
    ret.constraints.emplace_back(FromProto(pb_con));
  }
  for (const auto& pb_ref : block.refs()) {
    Refinement ref;
    switch (pb_ref.dir()) {
      case proto::Refinement::None:
        ref.dir = RefDir::None;
        break;
      case proto::Refinement::In:
        ref.dir = RefDir::In;
        break;
      case proto::Refinement::Out:
        ref.dir = RefDir::Out;
        break;
      case proto::Refinement::InOut:
        ref.dir = RefDir::InOut;
        break;
      default:
        break;
    }
    ref.from = pb_ref.from();
    ref.into = pb_ref.into();
    for (const auto& pb_off : pb_ref.access()) {
      ref.access.emplace_back(FromProto(pb_off));
    }
    ref.shape = tile::FromProto(pb_ref.shape());
    ref.agg_op = pb_ref.agg_op();
    ret.refs.emplace_back(ref);
  }
  for (const auto& pb_stmt : block.stmts()) {
    switch (pb_stmt.op_case()) {
      case proto::Statement::kLoad:
        ret.stmts.emplace_back(std::make_shared<Load>(pb_stmt.load().from(), pb_stmt.load().into()));
        break;
      case proto::Statement::kStore:
        ret.stmts.emplace_back(std::make_shared<Store>(pb_stmt.store().from(), pb_stmt.store().into()));
        break;
      case proto::Statement::kConstant:
        switch (pb_stmt.constant().value_case()) {
          case proto::Constant::kIconst:
            ret.stmts.emplace_back(std::make_shared<Constant>(pb_stmt.constant().name(), pb_stmt.constant().iconst()));
            break;
          case proto::Constant::kFconst:
            ret.stmts.emplace_back(std::make_shared<Constant>(pb_stmt.constant().name(), pb_stmt.constant().fconst()));
            break;
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
        ret.stmts.emplace_back(stmt);
      } break;
      case proto::Statement::kIntrinsic: {
        auto stmt = std::make_shared<Intrinsic>();
        stmt->name = pb_stmt.intrinsic().name();
        for (const auto& item : pb_stmt.intrinsic().inputs()) {
          stmt->inputs.push_back(item);
        }
        for (const auto& item : pb_stmt.intrinsic().outputs()) {
          stmt->outputs.push_back(item);
        }
        ret.stmts.emplace_back(stmt);
      } break;
      case proto::Statement::kBlock: {
        auto stmt = std::make_shared<Block>();
        *stmt = FromProto(pb_stmt.block());
        ret.stmts.emplace_back(stmt);
      } break;
      default:
        break;
    }
  }
  for (const auto& item : block.annotations()) {
    if (item.second.type_url() == "type.vertex.ai/vertexai.tile.stripe.proto.BoolAnnotation") {
      proto::BoolAnnotation pb_ann;
      item.second.UnpackTo(&pb_ann);
      auto ann = std::make_shared<BoolAnnotation>(pb_ann.value());
      ret.annotations.insert(std::make_pair(item.first, ann));
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

proto::Block IntoProto(const Block& block) {
  proto::Block ret;
  ret.set_name(block.name);
  ret.set_comments(block.comments);
  for (const auto& idx : block.idxs) {
    auto pb_idx = ret.add_idxs();
    pb_idx->set_name(idx.name);
    pb_idx->set_range(idx.range);
    pb_idx->set_factor(idx.factor);
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
    *pb_ref->mutable_shape() = IntoProto(ref.shape);
    pb_ref->set_agg_op(ref.agg_op);
  }
  for (const auto& stmt : block.stmts) {
    auto pb_stmt = ret.add_stmts();
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
  for (const auto& item : block.annotations) {
    google::protobuf::Any any;
    if (auto ann = BoolAnnotation::Downcast(item.second)) {
      proto::BoolAnnotation pb_ann;
      pb_ann.set_value(ann->value);
      any.PackFrom(pb_ann);
    }
    (*ret.mutable_annotations())[item.first] = any;
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

std::vector<Refinement>::iterator Block::ref_by_into(const std::string& name) {
  return std::find_if(refs.begin(), refs.end(), [&name](const Refinement& ref) { return ref.into == name; });
}

std::vector<Refinement>::const_iterator Block::ref_by_into(const std::string& name) const {
  return std::find_if(refs.begin(), refs.end(), [&name](const Refinement& ref) { return ref.into == name; });
}

std::string Block::unique_ref_name(const std::string& in) {
  if (ref_by_into(in) == refs.end()) {
    return in;
  }
  size_t i = 0;
  while (true) {
    std::string name = in + "_" + std::to_string(i++);
    if (ref_by_into(name) == refs.end()) {
      return name;
    }
  }
  // Unreachable
  return "";
}

Affine Refinement::FlatAccess() const {
  assert(access.size() == shape.dims.size());
  Affine ret;
  for (size_t i = 0; i < access.size(); i++) {
    ret += shape.dims[i].stride * access[i];
  }
  return ret;
}

}  // namespace stripe
}  // namespace tile
}  // namespace vertexai
