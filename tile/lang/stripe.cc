#include "tile/lang/stripe.h"

#include <sstream>

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

std::ostream& operator<<(std::ostream& os, const stripe::proto::RefineIn& ref) {
  os << ref.name() << ref.access();
  return os;
}

std::ostream& operator<<(std::ostream& os, const stripe::proto::RefineOut& ref) {
  os << ref.name() << ref.access();
  if (!ref.agg_op().empty()) {
    os << ":" << ref.agg_op();
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const stripe::proto::Constraint& constraint) {  //
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
    case stripe::proto::Statement::kPrimitive:
      PrintTab(os, depth);
      os << stmt.primitive() << std::endl;
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

std::ostream& operator<<(std::ostream& os, const stripe::proto::Primitive& op) {
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

std::ostream& operator<<(std::ostream& os, const stripe::proto::BufferAccess& access) {
  if (access.offset()) {
    os << ":" << access.offset();
  }
  os << "[";
  for (size_t j = 0; j < access.strides_size(); j++) {
    const auto& stride = access.strides(j);
    if (j > 0) {
      os << ", ";
    }
    os << stride;
  }
  os << "]";
  return os;
}

void Print(std::ostream& os, const stripe::proto::Block& block, size_t depth) {
  PrintTab(os, depth);
  os << "def " << block.name() << "[";
  for (size_t i = 0; i < block.index_names_size(); i++) {
    if (i > 0) {
      os << ", ";
    }
    os << block.index_names(i) << ":" << block.index_ranges(i);
  }
  os << "] (";
  for (size_t i = 0; i < block.ref_ins_size(); i++) {
    const auto& input = block.ref_ins(i);
    if (i > 0) {
      os << ", ";
    }
    os << input;
  }
  os << ") -> (";
  for (size_t i = 0; i < block.ref_outs_size(); i++) {
    const auto& output = block.ref_outs(i);
    if (i > 0) {
      os << ", ";
    }
    os << output;
  }
  os << "):" << std::endl;
  if (!block.comments().empty()) {
    std::stringstream ss(block.comments());
    for (size_t i = 0; i < 2; i++) {
      std::string line;
      std::getline(ss, line, '\n');
      if (i > 0) {
        PrintTab(os, depth + 1);
        os << "// " << line << std::endl;
      }
    }
  }
  for (const auto& decl : block.decls()) {
    PrintTab(os, depth + 1);
    os << decl << std::endl;
  }
  for (const auto& stmt : block.stmts()) {
    Print(os, stmt, depth + 1);
  }
  // PrintTab(os, depth);
  // os << "}" << std::endl;
}

}  // namespace lang
}  // namespace tile
}  // namespace vertexai
