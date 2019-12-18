//===- WrapperGen.h - MLIR op lib dialect wrapper generator for C++ -------===//
//
// Copyright 2019 Intel Corporation.
//
// =============================================================================
//
// WrapperGen in C++ generates a file that wraps the C++ op lib with a fluent
// interface.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <vector>

#include "pmlc/dialect/op_lib/OpModel.h"

using llvm::raw_ostream;
using llvm::RecordKeeper;

namespace pmlc::dialect::op::tblgen::cpp {

namespace wrapper {

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

}  // namespace wrapper

static inline bool genWrappers(const RecordKeeper& recordKeeper, raw_ostream& os) {
  // First, grab all the data we'll ever need from the record and place it in a DialectInfo struct
  auto OpLibDialect = DialectInfo(recordKeeper);
  // Then, emit specifically for c++
  auto OpLibEmitter = wrapper::Emitter(OpLibDialect, os);
  return false;
}

}  // namespace pmlc::dialect::op::tblgen::cpp
