#include "tile/lang/stripe.h"

#include <sstream>

typedef google::protobuf::RepeatedField<::google::protobuf::int64> RepeatedInt64;
typedef google::protobuf::RepeatedPtrField<::std::string> RepeatedString;

namespace vertexai {
namespace tile {
namespace lang {

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

std::ostream& operator<<(std::ostream& os, const shape::proto::TensorShape::Dimension& dim) {  //
  os << dim.size() << ":" << dim.stride();
  return os;
}

std::ostream& operator<<(std::ostream& os, const shape::proto::TensorShape& shape) {  //
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

std::ostream& operator<<(std::ostream& os, const stripe::proto::Declaration& decl) {
  os << "var " << decl.name() << " : " << decl.shape();
  return os;
}

void Print(std::ostream& os, const stripe::proto::Statement& stmt, size_t depth) {
  switch (stmt.op_case()) {
    case stripe::proto::Statement::kLoad:
      PrintTab(os, depth);
      os << stmt.load() << std::endl;
      break;
    case stripe::proto::Statement::kStore:
      PrintTab(os, depth);
      os << stmt.store() << std::endl;
      break;
    case stripe::proto::Statement::kIntrinsic:
      PrintTab(os, depth);
      os << stmt.intrinsic() << std::endl;
      break;
    case stripe::proto::Statement::kSpecial:
      PrintTab(os, depth);
      os << stmt.special() << std::endl;
      break;
    case stripe::proto::Statement::kConstant:
      PrintTab(os, depth);
      os << stmt.constant() << std::endl;
      break;
    case stripe::proto::Statement::kBlock:
      Print(os, stmt.block(), depth);
      break;
    default:
      break;
  }
}

std::ostream& operator<<(std::ostream& os, const stripe::proto::Load& op) {
  os << op.into() << " = load(" << op.from() << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const stripe::proto::Store& op) {
  os << op.into() << " = store(" << op.from() << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const stripe::proto::Intrinsic& op) {
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

std::ostream& operator<<(std::ostream& os, const stripe::proto::Special& op) {
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

std::ostream& operator<<(std::ostream& os, const stripe::proto::Constant& op) {
  os << op.name() << " = ";
  switch (op.value_case()) {
    case stripe::proto::Constant::kIconst:
      os << op.iconst();
      break;
    case stripe::proto::Constant::kFconst:
      os << op.fconst();
      break;
    default:
      break;
  }
  return os;
}

void Print(std::ostream& os, const RepeatedInt64& strides, const RepeatedString& index_names, bool first = true) {
  for (size_t i = 0; i < strides.size(); i++) {
    const auto& stride = strides[i];
    if (stride) {
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
      if (i < index_names.size()) {
        os << index_names[i];
      } else {
        os << "#" << i;
      }
      first = false;
    }
  }
}

void Print(std::ostream& os, const stripe::proto::BufferAccess& access, const stripe::proto::Block& block) {
  if (access.offset() == 0 && access.strides_size() == 0) {
    return;
  }
  os << "[";
  bool first = true;
  if (access.offset()) {
    os << access.offset();
    first = false;
  }
  Print(os, access.strides(), block.index_names(), first);
  os << "]";
}

void Print(std::ostream& os, const stripe::proto::Constraint& constraint, const stripe::proto::Block& block) {
  Print(os, constraint.lhs(), block.index_names());
  os << " < " << constraint.rhs();
}

void Print(std::ostream& os, const stripe::proto::Block& block, size_t depth) {
  PrintTab(os, depth);
  os << "block [";
  for (size_t i = 0; i < block.index_names_size(); i++) {
    if (i > 0) {
      os << ", ";
    }
    os << block.index_names(i) << ":" << block.index_ranges(i);
  }
  os << "]";
  if (!block.name().empty()) {
    os << " // " << block.name();
  }
  os << std::endl;

  if (!block.comments().empty()) {
    std::stringstream ss(block.comments());
    for (size_t i = 0; i < 2; i++) {
      std::string line;
      std::getline(ss, line, '\n');
      if (i > 0) {
        PrintTab(os, depth + 2);
        os << "// " << line << std::endl;
      }
    }
  }
  for (const auto& constraint : block.constraints()) {
    PrintTab(os, depth + 2);
    Print(os, constraint, block);
    os << std::endl;
  }
  PrintTab(os, depth + 2);
  os << "(";
  for (size_t i = 0; i < block.ref_ins_size(); i++) {
    const auto& input = block.ref_ins(i);
    if (i > 0) {
      os << ", ";
    }
    os << input.name();
    Print(os, input.access(), block);
  }
  os << ") -> (";
  for (size_t i = 0; i < block.ref_outs_size(); i++) {
    const auto& output = block.ref_outs(i);
    if (i > 0) {
      os << ", ";
    }
    os << output.name();
    Print(os, output.access(), block);
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
    Print(os, stmt, depth + 1);
  }
  PrintTab(os, depth);
  os << "}" << std::endl;
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
