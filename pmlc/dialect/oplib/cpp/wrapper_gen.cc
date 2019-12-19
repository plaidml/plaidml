//===- WrapperGen.cc - MLIR op lib dialect wrapper generator for C++ ------===//
//
// Copyright 2019 Intel Corporation.
//
// =============================================================================
//
// WrapperGen in C++ generates a file that wraps the C++ op lib with a fluent
// interface.
//
//===----------------------------------------------------------------------===//

#include "pmlc/dialect/oplib/cpp/wrapper_gen.h"

#include <regex>
#include <vector>

#include "mlir/TableGen/Format.h"

#include "pmlc/dialect/oplib/cpp/utils.h"
#include "pmlc/dialect/oplib/model.h"

using llvm::raw_ostream;
using llvm::RecordKeeper;

namespace pmlc::dialect::oplib::cpp {

// The OpEmitter class is responsible for emitting the fluent EDSL code for each TableGen Record. It begins by
// querying the Record for relevant information about the Operator/Attributes/Results/Operatnds, then formats the
// information in in an EDSL-readable format.
class OpEmitter {
 private:
  OpInfo opInfo_;

 public:
  OpEmitter(const OpInfo& op, raw_ostream& os);
  void emitConstructor(raw_ostream& os);
  void emitDeclarations(raw_ostream& os);
  void emitOperatorOverload(raw_ostream& os);
  void emitSetters(raw_ostream& os);
};

class TypeEmitter {
 private:
  TypeInfo typeInfo_;

 public:
  TypeEmitter(const TypeInfo& type, raw_ostream& os);
};

class Emitter {
 private:
  DialectInfo info_;

 public:
  Emitter(DialectInfo info, raw_ostream& os);
  static void emitHeaders(raw_ostream& os);
  static void emitInits(raw_ostream& os);
  static void emitOps(const std::vector<OpInfo>& ops, raw_ostream& os);
  static void emitTypes(const std::vector<TypeInfo>& types, raw_ostream& os);
};

OpEmitter::OpEmitter(const OpInfo& op, raw_ostream& os) : opInfo_(OpInfo(op)) {
  os << formatv(commentHeader, opInfo_.name_, "wrapper");
  os << "class " << opInfo_.name_ << " {\n";
  os.indent(2) << "protected:\n";
  // Emit operand and attribute declarations
  emitDeclarations(os);
  os.indent(2) << "private:\n";
  // Emit constructor
  emitConstructor(os);
  // Emit setters
  emitSetters(os);
  // Emit operator overload
  emitOperatorOverload(os);
  os << "};\n";
}

void OpEmitter::emitConstructor(raw_ostream& os) {
  // Declare the types of parameters that must be passed into the constructor. Get this from the operands.
  if (opInfo_.operands_.size() == 1) {
    os.indent(4) << "explicit " << opInfo_.name_ << "(";
  } else {
    os.indent(4) << opInfo_.name_ << "(";
  }
  bool visited = false;
  for (const auto& operandPair : opInfo_.operands_) {
    if (visited) os << ", ";
    visited = true;
    os << convertType(operandPair.second) << " " << operandPair.first;
  }
  os << ") : ";
  // Initialize the private variables in the op class with the parameters passed in
  visited = false;
  for (const auto& operandPair : opInfo_.operands_) {
    if (visited) os << ", ";
    visited = true;
    os << operandPair.first << "_(" << operandPair.first << ")";
  }
  os << " {}\n";
}

void OpEmitter::emitDeclarations(raw_ostream& os) {
  for (const auto& operandPair : opInfo_.operands_) {
    os.indent(4) << convertType(operandPair.second) << " " << operandPair.first << "_;\n";
  }
  for (const auto& attributePair : opInfo_.attributes_) {
    os.indent(4) << convertType(attributePair.second) << " " << attributePair.first << "_;\n";
  }
}

void OpEmitter::emitOperatorOverload(raw_ostream& os) {
  os.indent(4) << "operator " << convertType(opInfo_.returnType_) << "() const {\n";
  os.indent(6) << "auto args = edsl::make_tuple(";
  bool visited = false;
  for (const auto& operandPair : opInfo_.operands_) {
    if (visited) os << ", ";
    visited = true;
    os << operandPair.first << "_";
  }
  for (const auto& attrPair : opInfo_.attributes_) {
    if (visited) os << ", ";
    visited = true;
    // special cases: TODO(perhaps try to get rid of and/or simplify these)
    bool is_enum = false;
    bool is_vector = false;
    std::regex enum_regex("PML.*Attr");
    if (std::regex_match(attrPair.second.str(), enum_regex)) {
      is_enum = true;
    } else if (attrPair.second.str() == "ArrayAttr") {
      is_vector = true;
    }
    if (is_enum) {
      os << "static_cast<int64_t>(";
    } else if (is_vector) {
      os << "edsl::make_tuple(";
    }
    os << attrPair.first << "_";
    if (is_enum || is_vector) {
      os << ")";
    }
  }
  os << ");\n";
  os.indent(6) << "return details::op(\"" << opInfo_.name_ << "\", args)";
  if (convertType(opInfo_.returnType_) == "edsl::Tensor") {
    os << ".as_tensor();";
  }
  os << "\n";
  os.indent(4) << "}\n";
}

void OpEmitter::emitSetters(raw_ostream& os) {
  // Setters must exist for each attribute.
  for (const auto& attributePair : opInfo_.attributes_) {
    os.indent(4) << opInfo_.name_ << " " << attributePair.first << "(" << convertType(attributePair.second) << " "
                 << attributePair.first << ") {\n";
    os.indent(6) << attributePair.first << "_ = " << attributePair.first << ";\n";
    os.indent(6) << "return *this;\n";
    os.indent(4) << "}\n";
  }
}

TypeEmitter::TypeEmitter(const TypeInfo& type, raw_ostream& os) : typeInfo_(TypeInfo(type)) {
  os << formatv(commentHeader, typeInfo_.name_, "enumerator");
  os << "enum class " << typeInfo_.name_ << " : " << convertType(typeInfo_.returnType_) << " { \n";
  for (const auto& optPair : typeInfo_.opts_) {
    os.indent(2) << optPair.first << " = " << optPair.second << ",\n";
  }
  os << "};\n\n";
}

Emitter::Emitter(DialectInfo info, raw_ostream& os) : info_(info) {
  emitHeaders(os);
  os << "namespace plaidml {\n"
     << "namespace op {\n\n";
  emitInits(os);
  emitTypes(info.all_types_, os);
  emitOps(info.all_ops_, os);
  os << "\n\n} // namespace op\n"
     << "} // namespace plaidml\n";
}

void Emitter::emitHeaders(raw_ostream& os) { os << fileCommentHeader << includeHeader; }

void Emitter::emitInits(raw_ostream& os) { os << initFunction << ffiFunction; }

void Emitter::emitOps(const std::vector<OpInfo>& ops, raw_ostream& os) {
  for (auto op : ops) {
    OpEmitter(op, os);
  }
}

void Emitter::emitTypes(const std::vector<TypeInfo>& types, raw_ostream& os) {
  for (auto type : types) {
    TypeEmitter(type, os);
  }
}

bool genWrappers(const RecordKeeper& recordKeeper, raw_ostream& os) {
  // First, grab all the data we'll ever need from the record and place it in a DialectInfo struct
  auto OpLibDialect = DialectInfo(recordKeeper);
  // Then, emit specifically for c++
  Emitter(OpLibDialect, os);
  return false;
}

}  // namespace pmlc::dialect::oplib::cpp
