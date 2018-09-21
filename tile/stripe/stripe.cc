#include "tile/stripe/stripe.h"

#include <sstream>

#include "base/util/printstring.h"

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
  // TODO
  return ret;
}

proto::Block IntoProto(const Block& block) {
  proto::Block ret;
  // TODO
  return ret;
}

}  // namespace stripe
}  // namespace tile
}  // namespace vertexai
