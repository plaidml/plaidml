//===- OpLibWrapperGen.cpp - MLIR op lib wrapper generator ----------------===//
//
// Copyright 2019 Intel Corporation.
//
// =============================================================================
//
// OpLibWrapperGen
//
//===----------------------------------------------------------------------===//

#include <regex>

#include "mlir/TableGen/Format.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/OpInterfaces.h"
#include "llvm/Support/Signals.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#include "pmlc/dialect/op_lib/cpp/ImplGen.h"

using llvm::MapVector;
using llvm::raw_ostream;
using llvm::Record;
using llvm::RecordKeeper;
using mlir::GenRegistration;
using mlir::StringRef;
using mlir::tblgen::EnumAttr;
using mlir::tblgen::Operator;

namespace pmlc::dialect::op::tblgen {

using namespace pmlc::dialect::op;  // NOLINT [build/namespaces]

namespace cpp {

namespace impl {

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
  emitTypes(info.all_types_, os);
  os << "\n\n} // namespace op\n"
     << "} // namespace plaidml\n";
}
void Emitter::emitHeaders(raw_ostream& os) { os << fileCommentHeader; }
void Emitter::emitTypes(const std::vector<TypeInfo>& types, raw_ostream& os) {
  for (auto type : types) {
    TypeEmitter(type, os);
  }
}

}  // namespace impl

}  // namespace cpp
}  // namespace pmlc::dialect::op::tblgen
