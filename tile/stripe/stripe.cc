#include "tile/stripe/stripe.h"

#include <sstream>

#include "base/util/printstring.h"
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

void PrintBlock(std::ostream& os, const Block& block, size_t depth);

void PrintTab(std::ostream& os, size_t depth) {  //
  os << std::string(depth * 2, ' ');
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

void PrintStatement(std::ostream& os, const std::shared_ptr<Statement>& stmt, size_t depth) {
  switch (stmt->kind()) {
    case StmtKind::Load:
      PrintTab(os, depth);
      os << *std::dynamic_pointer_cast<Load>(stmt) << std::endl;
      break;
    case StmtKind::Store:
      PrintTab(os, depth);
      os << *std::dynamic_pointer_cast<Store>(stmt) << std::endl;
      break;
    case StmtKind::Intrinsic:
      PrintTab(os, depth);
      os << *std::dynamic_pointer_cast<Intrinsic>(stmt) << std::endl;
      break;
    case StmtKind::Special:
      PrintTab(os, depth);
      os << *std::dynamic_pointer_cast<Special>(stmt) << std::endl;
      break;
    case StmtKind::Constant:
      PrintTab(os, depth);
      os << *std::dynamic_pointer_cast<Constant>(stmt) << std::endl;
      break;
    case StmtKind::Block:
      PrintBlock(os, *std::dynamic_pointer_cast<Block>(stmt), depth);
      break;
    default:
      break;
  }
}

void PrintStride(std::ostream& os, int64_t stride, const std::string& name, bool first) {
  if (stride < 1) {
    if (first) {
      os << "-";
    } else {
      os << " - ";
    }
  } else if (!first) {
    os << " + ";
  }
  if (std::abs(stride) != 1) {
    os << std::abs(stride) << "*";
  }
  os << name;
}

void PrintStrides(std::ostream& os, const std::vector<int64_t>& strides, const std::vector<Index>& idxs, bool first) {
  // assert(strides.size() == idxs.size());
  for (size_t i = 0; i < strides.size(); i++) {
    const auto& stride = strides[i];
    if (stride) {
      std::string name;
      if (i < idxs.size()) {
        name = idxs[i].name;
      } else {
        name = printstring("#%zu", i);
      }
      PrintStride(os, stride, name, first);
      first = false;
    }
  }
}

void PrintAccess(std::ostream& os, const BufferAccess& access, const Block& block) {
  if (access.offset == 0 && access.strides.size() == 0) {
    return;
  }
  os << "[";
  bool first = true;
  if (access.offset) {
    os << access.offset;
    first = false;
  }
  PrintStrides(os, access.strides, block.idxs, first);
  os << "]";
}

void PrintConstraint(std::ostream& os, const Constraint& constraint, const Block& block) {
  PrintStrides(os, constraint.lhs, block.idxs, true);
  os << " < " << constraint.rhs;
}

void PrintBlock(std::ostream& os, const Block& block, size_t depth) {
  PrintTab(os, depth);
  os << "block [";
  for (size_t i = 0; i < block.idxs.size(); i++) {
    if (i > 0) {
      os << ", ";
    }
    os << block.idxs[i].name << ":" << block.idxs[i].range;
  }
  os << "]";
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
    PrintConstraint(os, constraint, block);
    os << std::endl;
  }
  PrintTab(os, depth + 2);
  os << "(";
  for (size_t i = 0; i < block.refs.size(); i++) {
    const auto& ref = block.refs[i];
    if (i > 0) {
      os << ", ";
    }
    switch (ref.dir) {
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
    os << ref.into;
    if (ref.into != ref.from) {
      os << " = " << ref.from;
    }
    PrintAccess(os, ref.access, block);
    if (!ref.agg_op.empty()) {
      os << ":" << ref.agg_op;
    }
  }
  os << ") {" << std::endl;
  for (const auto& decl : block.decls) {
    PrintTab(os, depth + 1);
    os << "var " << decl.first << " : " << decl.second << std::endl;
  }
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

std::vector<Refinement> Block::ref_ins() const {
  std::vector<Refinement> results;
  for (const auto& ref : refs) {
    if (ref.dir == RefDir::In) {
      results.push_back(ref);
    }
  }
  return results;
}

std::vector<Refinement> Block::ref_outs() const {
  std::vector<Refinement> results;
  for (const auto& ref : refs) {
    if (ref.dir == RefDir::Out) {
      results.push_back(ref);
    }
  }
  return results;
}

std::ostream& operator<<(std::ostream& os, const BufferAccess& access) {
  os << access.offset << ":"
     << "TODO";
  return os;
}

bool operator==(const BufferAccess& lhs, const BufferAccess& rhs) {
  return std::tie(lhs.offset, lhs.strides) ==  //
         std::tie(rhs.offset, rhs.strides);
}

bool operator==(const Index& lhs, const Index& rhs) {
  return std::tie(lhs.name, lhs.range, lhs.factor) ==  //
         std::tie(rhs.name, rhs.range, rhs.factor);
}

bool operator==(const Constraint& lhs, const Constraint& rhs) {
  return std::tie(lhs.lhs, lhs.rhs) ==  //
         std::tie(rhs.lhs, rhs.rhs);
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
  for (const auto& pb_cons : block.constraints()) {
    Constraint cons;
    for (const auto& pb_lhs : pb_cons.lhs()) {
      cons.lhs.push_back(pb_lhs);
    }
    cons.rhs = pb_cons.rhs();
    ret.constraints.emplace_back(cons);
  }
  for (const auto& pb_decl : block.decls()) {
    ret.decls.insert(std::make_pair(pb_decl.first, tile::FromProto(pb_decl.second)));
  }
  for (const auto& pb_ref : block.refs()) {
    Refinement ref;
    switch (pb_ref.dir()) {
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
    ref.access.offset = pb_ref.access().offset();
    for (const auto& stride : pb_ref.access().strides()) {
      ref.access.strides.push_back(stride);
    }
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
  for (const auto& cons : block.constraints) {
    auto pb_cons = ret.add_constraints();
    for (const auto& lhs : cons.lhs) {
      pb_cons->add_lhs(lhs);
    }
    pb_cons->set_rhs(cons.rhs);
  }
  for (const auto& decl : block.decls) {
    (*ret.mutable_decls())[decl.first] = IntoProto(decl.second);
  }
  for (const auto& ref : block.refs) {
    auto pb_ref = ret.add_refs();
    switch (ref.dir) {
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
    auto pb_access = pb_ref->mutable_access();
    pb_access->set_offset(ref.access.offset);
    for (const auto& stride : ref.access.strides) {
      pb_access->add_strides(stride);
    }
    pb_ref->set_agg_op(ref.agg_op);
  }
  for (const auto& stmt : block.stmts) {
    auto pb_stmt = ret.add_stmts();
    switch (stmt->kind()) {
      case StmtKind::Load: {
        auto load = std::dynamic_pointer_cast<Load>(stmt);
        auto pb_load = pb_stmt->mutable_load();
        pb_load->set_from(load->from);
        pb_load->set_into(load->into);
      } break;
      case StmtKind::Store: {
        auto store = std::dynamic_pointer_cast<Store>(stmt);
        auto pb_store = pb_stmt->mutable_store();
        pb_store->set_from(store->from);
        pb_store->set_into(store->into);
      } break;
      case StmtKind::Constant: {
        auto constant = std::dynamic_pointer_cast<Constant>(stmt);
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
        auto special = std::dynamic_pointer_cast<Special>(stmt);
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
        auto intrinsic = std::dynamic_pointer_cast<Intrinsic>(stmt);
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
        auto inner = std::dynamic_pointer_cast<Block>(stmt);
        *pb_stmt->mutable_block() = IntoProto(*inner);
      } break;
    }
  }
  for (const auto& item : block.annotations) {
    google::protobuf::Any any;
    if (auto ann = std::dynamic_pointer_cast<BoolAnnotation>(item.second)) {
      proto::BoolAnnotation pb_ann;
      pb_ann.set_value(ann->value);
      any.PackFrom(pb_ann);
    }
    (*ret.mutable_annotations())[item.first] = any;
  }
  return ret;
}

}  // namespace stripe
}  // namespace tile
}  // namespace vertexai
