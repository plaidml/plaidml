#include "tile/lang/stripe.h"

#include <sstream>

#include "base/util/printstring.h"

namespace vertexai {
namespace tile {
namespace stripe {
namespace proto {

typedef google::protobuf::RepeatedField<google::protobuf::int64> RepeatedInt64;
typedef google::protobuf::RepeatedPtrField<Index> RepeatedIndex;

void PrintTab(std::ostream& os, size_t depth) {  //
  os << std::string(depth * 2, ' ');
}

std::ostream& operator<<(std::ostream& os, const shape::proto::TensorShape::DataType& type) {  //
  switch (type) {
    case shape::proto::TensorShape::BOOLEAN:
      os << "BOOL";
      break;
    case shape::proto::TensorShape::INT8:
      os << "INT8";
      break;
    case shape::proto::TensorShape::INT16:
      os << "INT16";
      break;
    case shape::proto::TensorShape::INT32:
      os << "INT32";
      break;
    case shape::proto::TensorShape::INT64:
      os << "INT64";
      break;
    case shape::proto::TensorShape::UINT8:
      os << "UINT8";
      break;
    case shape::proto::TensorShape::UINT16:
      os << "UINT16";
      break;
    case shape::proto::TensorShape::UINT32:
      os << "UINT32";
      break;
    case shape::proto::TensorShape::UINT64:
      os << "UINT64";
      break;
    case shape::proto::TensorShape::FLOAT16:
      os << "FLOAT16";
      break;
    case shape::proto::TensorShape::FLOAT32:
      os << "FLOAT32";
      break;
    case shape::proto::TensorShape::FLOAT64:
      os << "FLOAT64";
      break;
    default:
      os << "INVALID";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const shape::proto::TensorShape::Dimension& dim) {
  os << dim.size() << ":" << dim.stride();
  return os;
}

std::ostream& operator<<(std::ostream& os, const shape::proto::TensorShape& shape) {
  os << shape.type() << "[";
  for (size_t i = 0; i < shape.dimensions_size(); i++) {
    if (i > 0) {
      os << ", ";
    }
    os << shape.dimensions(i);
  }
  os << "]";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Declaration& decl) {
  os << "var " << decl.name() << " : " << decl.shape();
  return os;
}

void PrintStatement(std::ostream& os, const Statement& stmt, size_t depth) {
  switch (stmt.op_case()) {
    case Statement::kLoad:
      PrintTab(os, depth);
      os << stmt.load() << std::endl;
      break;
    case Statement::kStore:
      PrintTab(os, depth);
      os << stmt.store() << std::endl;
      break;
    case Statement::kIntrinsic:
      PrintTab(os, depth);
      os << stmt.intrinsic() << std::endl;
      break;
    case Statement::kSpecial:
      PrintTab(os, depth);
      os << stmt.special() << std::endl;
      break;
    case Statement::kConstant:
      PrintTab(os, depth);
      os << stmt.constant() << std::endl;
      break;
    case Statement::kBlock:
      PrintBlock(os, stmt.block(), depth);
      break;
    default:
      break;
  }
}

std::ostream& operator<<(std::ostream& os, const Load& op) {
  os << op.into() << " = load(" << op.from() << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Store& op) {
  os << op.into() << " = store(" << op.from() << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Intrinsic& op) {
  if (op.outputs_size() > 1) {
    os << "(";
  }
  for (size_t i = 0; i < op.outputs_size(); i++) {
    if (i > 0) {
      os << ", ";
    }
    os << op.outputs(i);
  }
  if (op.outputs_size() > 1) {
    os << ")";
  }
  os << " = " << op.name() << "(";
  for (size_t i = 0; i < op.inputs_size(); i++) {
    if (i > 0) {
      os << ", ";
    }
    os << op.inputs(i);
  }
  os << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Special& op) {
  if (op.outputs_size() > 1) {
    os << "(";
  }
  for (size_t i = 0; i < op.outputs_size(); i++) {
    if (i > 0) {
      os << ", ";
    }
    os << op.outputs(i);
  }
  if (op.outputs_size() > 1) {
    os << ")";
  }
  os << " = " << op.name() << "(";
  for (size_t i = 0; i < op.inputs_size(); i++) {
    if (i > 0) {
      os << ", ";
    }
    os << op.inputs(i);
  }
  os << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Constant& op) {
  os << op.name() << " = ";
  switch (op.value_case()) {
    case Constant::kIconst:
      os << op.iconst();
      break;
    case Constant::kFconst:
      os << op.fconst();
      break;
    default:
      break;
  }
  return os;
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

void PrintStrides(std::ostream& os, const RepeatedInt64& strides, const RepeatedIndex& idxs, bool first) {
  // assert(strides.size() == idxs.size());
  for (size_t i = 0; i < strides.size(); i++) {
    const auto& stride = strides[i];
    if (stride) {
      std::string name;
      if (i < idxs.size()) {
        name = idxs[i].name();
      } else {
        name = printstring("#%zu", i);
      }
      PrintStride(os, stride, name, first);
      first = false;
    }
  }
}

void PrintAccess(std::ostream& os, const BufferAccess& access, const Block& block) {
  if (access.offset() == 0 && access.strides_size() == 0) {
    return;
  }
  os << "[";
  bool first = true;
  if (access.offset()) {
    os << access.offset();
    first = false;
  }
  PrintStrides(os, access.strides(), block.idxs(), first);
  os << "]";
}

void PrintConstraint(std::ostream& os, const Constraint& constraint, const Block& block) {
  PrintStrides(os, constraint.lhs(), block.idxs(), true);
  os << " < " << constraint.rhs();
}

void PrintBlock(std::ostream& os, const Block& block, size_t depth) {
  PrintTab(os, depth);
  os << "block [";
  for (size_t i = 0; i < block.idxs_size(); i++) {
    if (i > 0) {
      os << ", ";
    }
    os << block.idxs(i).name() << ":" << block.idxs(i).range();
  }
  os << "]";
  if (!block.name().empty()) {
    os << " // " << block.name();
  }
  os << std::endl;

  if (!block.comments().empty()) {
    std::stringstream ss(block.comments());
    for (std::string line; std::getline(ss, line, '\n');) {
      PrintTab(os, depth + 2);
      os << "// " << line << std::endl;
    }
  }
  for (const auto& constraint : block.constraints()) {
    PrintTab(os, depth + 2);
    PrintConstraint(os, constraint, block);
    os << std::endl;
  }
  PrintTab(os, depth + 2);
  os << "(";
  for (size_t i = 0; i < block.ref_ins_size(); i++) {
    const auto& input = block.ref_ins(i);
    if (i > 0) {
      os << ", ";
    }
    os << input.into();
    if (input.into() != input.from()) {
      os << " = " << input.from();
    }
    PrintAccess(os, input.access(), block);
  }
  os << ") -> (";
  for (size_t i = 0; i < block.ref_outs_size(); i++) {
    const auto& output = block.ref_outs(i);
    if (i > 0) {
      os << ", ";
    }
    os << output.into();
    if (output.into() != output.from()) {
      os << " = " << output.from();
    }
    PrintAccess(os, output.access(), block);
    if (!output.agg_op().empty()) {
      os << ":" << output.agg_op();
    }
  }
  os << ") {" << std::endl;
  for (const auto& decl : block.decls()) {
    PrintTab(os, depth + 1);
    os << decl << std::endl;
  }
  for (const auto& stmt : block.stmts()) {
    PrintStatement(os, stmt, depth + 1);
  }
  PrintTab(os, depth);
  os << "}" << std::endl;
}

std::ostream& operator<<(std::ostream& os, const Block& block) {
  PrintBlock(os, block, 0);
  return os;
}

}  // namespace proto
}  // namespace stripe
}  // namespace tile
}  // namespace vertexai
