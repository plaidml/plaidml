//===- OpLibWrapperGen.cpp - MLIR op lib wrapper generator ----------------===//
//
// Copyright 2019 Intel Corporation.
//
// =============================================================================
//
// OpLibWrapperGen
//
//===----------------------------------------------------------------------===//

#include <map>

#include "mlir/Support/STLExtras.h"
#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/OpInterfaces.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

using llvm::raw_ostream;
using llvm::Record;
using llvm::RecordKeeper;
using mlir::StringRef;
using mlir::tblgen::Operator;

static const char* TBLGEN_DEBUG = std::getenv("TBLGEN_DEBUG");

static const char* const opCommentHeader = R"(
//===----------------------------------------------------------------------===//
// {0} {1}
//===----------------------------------------------------------------------===//

)";

static inline const std::map<StringRef, StringRef> typeLookupTable = {
    {"APInt", "int"},
    {"ArrayAttr", "std::vector<int>"},
    {"bool", "bool"},
    {"StringRef", "std::string"},
    {"tensor of any type values", "edsl::Tensor"},
    {"vector of integer values of length 2", "std::vector<int>"},
};

static StringRef convertTypeToEDSL(const StringRef type) {
  StringRef edslType = "unrecognized";
  auto found = typeLookupTable.find(type);
  if (found != typeLookupTable.end()) {
    edslType = found->second;
  }
  return edslType;
}

// The OpFluidEmitter class is responsible for emitting the fluent EDSL code for each TableGen Record. It begins by
// querying the Record for relevant information about the Operator/Attributes/Results/Operatnds, then formats the
// information in in an EDSL-readable format.
class OpFluidEmitter {
 private:
  StringRef name_;
  StringRef returnType_;
  std::map<StringRef, StringRef> attributes_;
  std::map<StringRef, StringRef> operands_;

 public:
  inline OpFluidEmitter(const Operator& op, raw_ostream& os) {
    name_ = op.getCppClassName();
    returnType_ = getReturnType(op);
    getAttributes(op);
    getOperands(op);
    os << "class " << name_ << " {\n";
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

  inline void emitConstructor(raw_ostream& os) {
    // Declare the types of parameters that must be passed into the constructor. Get this from the operands.
    os.indent(4) << "explicit " << name_ << "(";
    bool visited = false;
    for (const auto& operandPair : operands_) {
      if (visited) os << ", ";
      visited = true;
      os << operandPair.second << " " << operandPair.first;
    }
    os << ") : ";
    // Initialize the private variables in the op class with the parameters passed in
    visited = false;
    for (const auto& operandPair : operands_) {
      if (visited) os << ", ";
      visited = true;
      os << operandPair.first << "_(" << operandPair.first << ")";
    }
    os << " {}\n";
  }

  inline void emitDeclarations(raw_ostream& os) {
    for (const auto& operandPair : operands_) {
      os.indent(4) << operandPair.second << " " << operandPair.first << "_;\n";
    }
    for (const auto& attributePair : attributes_) {
      os.indent(4) << attributePair.second << " " << attributePair.first << "_;\n";
    }
  }

  inline void emitOperatorOverload(raw_ostream& os) {
    os.indent(4) << "operator " << returnType_ << "() const {\n";
    os.indent(6) << "auto args = edsl::make_tuple();\n";
    os.indent(6) << "return details::op(\"" << name_ << "\", args)";
    if (returnType_ == "edsl::Tensor") {
      os << ".as_tensor();";
    }
    os << "\n";
    os.indent(4) << "}\n";
  }

  inline void emitSetters(raw_ostream& os) {
    // Setters must exist for each attribute.
    for (const auto& attributePair : attributes_) {
      os.indent(4) << name_ << " " << attributePair.first << "(" << attributePair.second << " " << attributePair.first
                   << ") {\n";
      os.indent(6) << attributePair.first << "_ = " << attributePair.first << ";\n";
      os.indent(6) << "return *this;\n";
      os.indent(4) << "}\n";
    }
  }

  inline void getAttributes(const Operator& op) {
    for (auto& namedAttr : op.getAttributes()) {
      const auto& name = namedAttr.name;
      const auto& mlir_type = namedAttr.attr.getReturnType();
      const StringRef edsl_type = convertTypeToEDSL(mlir_type);
      attributes_.insert({name, edsl_type});
    }
  }

  inline void getOperands(const Operator& op) {
    for (int index = 0; index < op.getNumOperands(); index++) {
      auto& namedOperand = op.getOperand(index);
      const auto& name = namedOperand.name;
      const auto& mlir_type = namedOperand.constraint.getDescription();
      const StringRef edsl_type = convertTypeToEDSL(mlir_type);
      operands_.insert({name, edsl_type});
    }
  }

  static inline StringRef getReturnType(const Operator& op) {
    StringRef edsl_type;
    int n_results = op.getNumResults();
    if (n_results > 1) {
      edsl_type = "edsl::Value&";
    } else if (n_results == 0) {
      edsl_type = "void";
    } else {
      auto& namedResult = op.getResult(0);
      const auto& mlir_type = namedResult.constraint.getDescription();
      edsl_type = convertTypeToEDSL(mlir_type);
    }
    return edsl_type;
  }
};

static void emitOpLibHeaders(raw_ostream& os) {
  os << "#pragma once\n\n"
     << "#include <string>\n"
     << "#include <vector>\n\n"
     << "#include \"plaidml2/edsl/edsl.h\"\n"
     << "#include \"plaidml2/op/ffi.h\"\n\n";
}

static void emitOpClasses(const std::vector<Record*>& defs, raw_ostream& os) {
  for (auto* def : defs) {
    Operator op(*def);
    os << formatv(opCommentHeader, op.getCppClassName(), "wrapper");
    OpFluidEmitter(op, os);
  }
}

static bool emitOpLibWrappers(const RecordKeeper& recordKeeper, raw_ostream& os) {
  emitSourceFileHeader("Op Lib EDSL Wrapper", os);
  emitOpLibHeaders(os);
  // inits, op definition
  os << "namespace plaidml {\n"
     << "namespace op {\n\n"
     << "static const char* const NCX = \"ncx\";\n"
     << "static const char* const NXC = \"nxc\";\n"
     << "static const char* const KCX = \"kcx\";\n"
     << "static const char* const XCK = \"xck\";\n\n"
     << "inline void init() {  //\n"
     << "  plaidml::init();\n"
     << "  plaidml::edsl::init();\n"
     << "  ffi::call_void(plaidml_op_init);\n"
     << "}\n\n"
     << "namespace details {\n\n"
     << "inline edsl::Value op(const std::string& name, const edsl::Value& args) {\n"
     << "  return edsl::Value(ffi::call<plaidml_expr*>(plaidml_op_make, name.c_str(), args.as_ptr()));\n"
     << "}\n\n"
     << "} // namespace details\n\n";
  const auto& defs = recordKeeper.getAllDerivedDefinitions("Op");
  emitOpClasses(defs, os);
  os << "\n\n} // namespace op\n"
     << "} // namespace plaidml\n";
  return false;
}

static mlir::GenRegistration genOpCPPs("gen-op-lib-wrappers", "Generate op lib EDSL wrappers",
                                       [](const RecordKeeper& records, raw_ostream& os) {
                                         return emitOpLibWrappers(records, os);
                                       });
