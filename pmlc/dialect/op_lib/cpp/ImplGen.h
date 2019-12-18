//===- ImplGen.h - MLIR op lib dialect implementation generator for C++ ---===//
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

#pragma once

#include <vector>

#include "pmlc/dialect/op_lib/cpp/utils.h"

#include "pmlc/dialect/op_lib/OpModel.h"

using llvm::raw_ostream;
using llvm::RecordKeeper;

namespace pmlc::dialect::op::tblgen::cpp {

namespace impl {

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
  static void emitTypes(const std::vector<TypeInfo>& types, raw_ostream& os);
};

}  // namespace impl

static inline bool genImpl(const RecordKeeper& recordKeeper, raw_ostream& os) {
  // First, grab all the data we'll ever need from the record and place it in a DialectInfo struct
  auto OpLibDialect = DialectInfo(recordKeeper);
  // Then, emit specifically for c++
  auto OpLibEmitter = impl::Emitter(OpLibDialect, os);
  return false;
}

}  // namespace pmlc::dialect::op::tblgen::cpp
