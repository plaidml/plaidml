//===- ImplGen.cc - MLIR op lib dialect implementation generator for C++ --===//
//
// Copyright 2019 Intel Corporation.
//
// =============================================================================
//
// ImplGen for C++ generates a file that C++ users can include in their own
// code to fulfill the dependencies needed for the C++ op lib, i.e. custom
// types.
//
//===----------------------------------------------------------------------===//

#include <regex>

#include "mlir/TableGen/Format.h"

#include "pmlc/dialect/op_lib/cpp/ImplGen.h"

using llvm::raw_ostream;

namespace pmlc::dialect::op::tblgen::cpp::impl {

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

}  // namespace pmlc::dialect::op::tblgen::cpp::impl
