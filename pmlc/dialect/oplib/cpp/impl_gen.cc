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

#include "pmlc/dialect/oplib/cpp/impl_gen.h"

#include <vector>

#include "mlir/TableGen/Format.h"

#include "pmlc/dialect/oplib/cpp/utils.h"
#include "pmlc/dialect/oplib/model.h"

using llvm::raw_ostream;
using llvm::RecordKeeper;

namespace pmlc::dialect::oplib::cpp {

class Emitter {
 private:
  DialectInfo* info_;
  raw_ostream& os_;

 public:
  Emitter(DialectInfo* info, raw_ostream& os) : info_(info), os_(os) {}

  void emit() {
    os_ << fileCommentHeader;
    os_ << "namespace plaidml {\n"
        << "namespace op {\n\n";
    emitTypes(info_->all_types_);
    os_ << "\n\n} // namespace op\n"
        << "} // namespace plaidml\n";
  }

  void emitTypes(const std::vector<TypeInfo>& types) {
    for (auto type : types) {
      emitType(type);
    }
  }

  void emitType(const TypeInfo& typeInfo) {
    os_ << formatv(commentHeader, typeInfo.name_, "enumerator");
    os_ << "enum class " << typeInfo.name_ << " : " << convertType(typeInfo.returnType_) << " { \n";
    for (const auto& optPair : typeInfo.opts_) {
      os_.indent(2) << optPair.first << " = " << optPair.second << ",\n";
    }
    os_ << "};\n\n";
  }
};

bool genImpl(const RecordKeeper& recordKeeper, raw_ostream& os) {
  // First, grab all the data we'll ever need from the record and place it in a DialectInfo struct
  auto info = DialectInfo(recordKeeper);
  // Then, emit specifically for c++
  Emitter(&info, os).emit();
  return false;
}

}  // namespace pmlc::dialect::oplib::cpp
