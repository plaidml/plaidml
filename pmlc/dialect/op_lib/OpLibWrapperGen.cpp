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

#include "base/util/logging.h"

#include "mlir/Support/STLExtras.h"
#include "mlir/TableGen/Attribute.h"
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
using mlir::GenRegistration;
using mlir::StringRef;
using mlir::tblgen::EnumAttr;
using mlir::tblgen::Operator;

namespace pmlc {
namespace dialect {
namespace op {

using namespace pmlc::dialect::op;  // NOLINT [build/namespaces]

namespace tblgen {

static inline std::map<StringRef, StringRef> getAttributes(const Operator& op) {
  std::map<StringRef, StringRef> attributes_;
  for (auto& namedAttr : op.getAttributes()) {
    const auto& name = namedAttr.name;
    auto mlir_type = namedAttr.attr.getReturnType();
    const auto& is_enum = namedAttr.attr.isEnumAttr();
    if (is_enum) {
      mlir_type = namedAttr.attr.getAttrDefName();
    }
    attributes_.insert({name, mlir_type});
  }
  return attributes_;
}

static inline std::map<StringRef, StringRef> getOperands(const Operator& op) {
  std::map<StringRef, StringRef> operands_;
  for (int index = 0; index < op.getNumOperands(); index++) {
    auto& namedOperand = op.getOperand(index);
    const auto& name = namedOperand.name;
    const auto& mlir_type = namedOperand.constraint.getDescription();
    operands_.insert({name, mlir_type});
  }
  return operands_;
}

static inline StringRef getReturnType(const Operator& op) {
  StringRef type;
  int n_results = op.getNumResults();
  if (n_results > 1) {
    type = "MLIR_LIST";
  } else if (n_results == 0) {
    type = "MLIR_VOID";
  } else {
    auto& namedResult = op.getResult(0);
    type = namedResult.constraint.getDescription();
  }
  return type;
}

struct OpInfo {
  StringRef name_;
  StringRef returnType_;
  std::map<StringRef, StringRef> attributes_;
  std::map<StringRef, StringRef> operands_;
  explicit OpInfo(const Operator& op) {
    name_ = op.getCppClassName();
    returnType_ = getReturnType(op);
    attributes_ = getAttributes(op);
    operands_ = getOperands(op);
  }
};

struct TypeInfo {
  StringRef name_;
  std::map<StringRef, int> opts_;
  StringRef returnType_;
  explicit TypeInfo(const EnumAttr& ea) {
    name_ = ea.getEnumClassName();
    returnType_ = ea.getUnderlyingType();
    for (auto eacase : ea.getAllCases()) {
      opts_.insert({eacase.getSymbol(), eacase.getValue()});
    }
  }
};

// TODO(dgkutnic): Make DialectInfo into a protobuf object.
struct DialectInfo {
  std::vector<OpInfo> all_ops_;
  std::vector<TypeInfo> all_types_;
  std::vector<OpInfo> getOpRecords(const RecordKeeper& recordKeeper) {
    std::vector<OpInfo> all_ops;
    auto pmlOpRecords = recordKeeper.getAllDerivedDefinitions("PML_Op");
    for (auto* record : pmlOpRecords) {
      all_ops.push_back(OpInfo(Operator(*record)));
    }
    return all_ops;
  }
  std::vector<TypeInfo> getTypeRecords(const RecordKeeper& recordKeeper) {
    std::vector<TypeInfo> all_types;
    auto enumTypeRecords = recordKeeper.getAllDerivedDefinitions("EnumAttrInfo");
    for (auto* record : enumTypeRecords) {
      all_types.push_back(TypeInfo(EnumAttr(*record)));
    }
    return all_types;
  }
  explicit DialectInfo(const RecordKeeper& recordKeeper)
      : all_ops_(getOpRecords(recordKeeper)), all_types_(getTypeRecords(recordKeeper)) {}
};

namespace cpp {

static const char* const fileCommentHeader = R"( // Copyright 2019 Intel Corporation.
/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Lib C++ EDSL Wrapper                                                    *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

)";

static const char* const includeHeader = R"(
#pragma once

#include <string>
#include <vector>

#include "plaidml2/edsl/edsl.h"
#include "plaidml2/op/ffi.h"

)";

static const char* const commentHeader = R"(
//===----------------------------------------------------------------------===//
// {0} {1}
//===----------------------------------------------------------------------===//

)";

static const char* const ffiFunction = R"(
namespace details {

inline edsl::Value op(const std::string& name, const edsl::Value& args) {
  return edsl::Value(ffi::call<plaidml_expr*>(plaidml_op_make, name.c_str(), args.as_ptr()));
}

} // namespace details

)";

static const char* const initFunction = R"(
inline void init() {  //
  plaidml::init();
  plaidml::edsl::init();
  ffi::call_void(plaidml_op_init);
}

)";

static inline const std::map<StringRef, StringRef> typeLookupTable = {
    {"APInt", "int"},
    {"ArrayAttr", "std::vector<int>"},
    {"bool", "bool"},
    {"MLIR_LIST", "edsl::Value&"},
    {"MLIR_VOID", "void"},
    {"PML_AutogroupModeAttr", "AutogroupMode"},
    {"PML_AutopadModeAttr", "AutopadMode"},
    {"PML_DerivModeAttr", "DerivMode"},
    {"PML_GroupLayoutAttr", "GroupLayout"},
    {"PML_TensorLayoutAttr", "TensorLayout"},
    {"StringRef", "std::string"},
    {"tensor of any type values", "edsl::Tensor"},
};

static StringRef convertType(const StringRef type) {
  StringRef edslType = "unrecognized";
  auto found = typeLookupTable.find(type);
  if (found != typeLookupTable.end()) {
    edslType = found->second;
  }
  return edslType;
}

// The OpEmitter class is responsible for emitting the fluent EDSL code for each TableGen Record. It begins by
// querying the Record for relevant information about the Operator/Attributes/Results/Operatnds, then formats the
// information in in an EDSL-readable format.
class OpEmitter {
 private:
  OpInfo opInfo_;

 public:
  inline OpEmitter(const OpInfo& op, raw_ostream& os) : opInfo_(OpInfo(op)) {
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

  inline void emitConstructor(raw_ostream& os) {
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

  inline void emitDeclarations(raw_ostream& os) {
    for (const auto& operandPair : opInfo_.operands_) {
      os.indent(4) << convertType(operandPair.second) << " " << operandPair.first << "_;\n";
    }
    for (const auto& attributePair : opInfo_.attributes_) {
      os.indent(4) << convertType(attributePair.second) << " " << attributePair.first << "_;\n";
    }
  }

  inline void emitOperatorOverload(raw_ostream& os) {
    os.indent(4) << "operator " << convertType(opInfo_.returnType_) << "() const {\n";
    os.indent(6) << "auto args = edsl::make_tuple();\n";
    os.indent(6) << "return details::op(\"" << opInfo_.name_ << "\", args)";
    if (convertType(opInfo_.returnType_) == "edsl::Tensor") {
      os << ".as_tensor();";
    }
    os << "\n";
    os.indent(4) << "}\n";
  }

  inline void emitSetters(raw_ostream& os) {
    // Setters must exist for each attribute.
    for (const auto& attributePair : opInfo_.attributes_) {
      os.indent(4) << opInfo_.name_ << " " << attributePair.first << "(" << convertType(attributePair.second) << " "
                   << attributePair.first << ") {\n";
      os.indent(6) << attributePair.first << "_ = " << attributePair.first << ";\n";
      os.indent(6) << "return *this;\n";
      os.indent(4) << "}\n";
    }
  }
};

class TypeEmitter {
 private:
  TypeInfo typeInfo_;

 public:
  inline TypeEmitter(const TypeInfo& type, raw_ostream& os) : typeInfo_(TypeInfo(type)) {
    os << formatv(commentHeader, typeInfo_.name_, "enumerator");
    os << "enum class " << typeInfo_.name_ << " : " << typeInfo_.returnType_ << " { \n";
    for (const auto& optPair : typeInfo_.opts_) {
      os.indent(2) << optPair.first << " = " << optPair.second << ",\n";
    }
    os << "};\n\n";
  }
};

class Emitter {
 private:
  DialectInfo info_;

 public:
  inline Emitter(DialectInfo info, raw_ostream& os) : info_(info) {
    emitHeaders(os);
    os << "namespace plaidml {\n"
       << "namespace op {\n\n";
    emitInits(os);
    emitTypes(info.all_types_, os);
    emitOps(info.all_ops_, os);
    os << "\n\n} // namespace op\n"
       << "} // namespace plaidml\n";
  }
  static void emitHeaders(raw_ostream& os) { os << fileCommentHeader << includeHeader; }
  static void emitInits(raw_ostream& os) { os << initFunction << ffiFunction; }
  static void emitOps(const std::vector<OpInfo>& ops, raw_ostream& os) {
    for (auto op : ops) {
      OpEmitter(op, os);
    }
  }
  static void emitTypes(const std::vector<TypeInfo>& types, raw_ostream& os) {
    for (auto type : types) {
      TypeEmitter(type, os);
    }
  }
};

static bool genWrappers(const RecordKeeper& recordKeeper, raw_ostream& os) {
  // First, grab all the data we'll ever need from the record and place it in a DialectInfo struct
  auto OpLibDialect = DialectInfo(recordKeeper);
  // Then, emit specifically for c++
  auto OpLibEmitter = Emitter(OpLibDialect, os);
  return false;
}

}  // namespace cpp

namespace python {

static const char* const fileCommentHeader = R"(## Copyright 2019 Intel Corporation.
##===- TableGen'erated file ----------------------------------*- Python -*-===##
##                                                                            ##
## Op Lib Python EDSL Wrapper                                                 ##
##                                                                            ##
## Automatically generated file, do not edit!                                 ##
##                                                                            ##
##===----------------------------------------------------------------------===##

)";

static const char* const includeHeader = R"(
import logging

import six

import plaidml2.edsl as edsl
from plaidml2.ffi import ffi, ffi_call, lib

)";

static const char* const commentHeader = R"(
##===----------------------------------------------------------------------===##
)";

static const char* const ffiFunction = R"(
def op(op_name, args):
    value = edsl.Value(args)
    return edsl.Value(ffi_call(lib.plaidml_op_make, op_name.encode(), value.as_ptr()))

)";

static const char* const initFunction = R"(
logger = logging.getLogger(__name__)


def __init():
    ffi_call(lib.plaidml_op_init)


ffi.init_once(__init, 'plaidml_op_init')

)";

class Emitter {
 private:
  DialectInfo info_;

 public:
  inline Emitter(DialectInfo info, raw_ostream& os) : info_(info) {
    emitHeaders(os);
    emitInits(os);
    emitOps(info.all_ops_, os);
  }
  static void emitHeaders(raw_ostream& os) { os << fileCommentHeader << includeHeader; }
  static void emitInits(raw_ostream& os) { os << initFunction << ffiFunction; }
  static void emitOps(const std::vector<OpInfo>& ops, raw_ostream& os) {
    for (auto op : ops) {
      os << commentHeader << "## " << op.name_ << commentHeader;
      // OpEmitter(op, os);
    }
  }
};

static bool genWrappers(const RecordKeeper& recordKeeper, raw_ostream& os) {
  // First, grab all the data we'll ever need from the record and place it in a DialectInfo struct
  auto OpLibDialect = DialectInfo(recordKeeper);
  // Then, emit specifically for python
  auto OpLibEmitter = Emitter(OpLibDialect, os);
  return false;
}

}  // namespace python

}  // namespace tblgen

static mlir::GenRegistration genOpCPPs("gen-op-lib-cpp-wrappers", "Generate Op Lib C++ EDSL wrappers",
                                       [](const RecordKeeper& records, raw_ostream& os) {
                                         return tblgen::cpp::genWrappers(records, os);
                                       });

static mlir::GenRegistration genOpPYs("gen-op-lib-py-wrappers", "Generate Op Lib Python EDSL wrappers",
                                      [](const RecordKeeper& records, raw_ostream& os) {
                                        return tblgen::python::genWrappers(records, os);
                                      });

}  // namespace op
}  // namespace dialect
}  // namespace pmlc
