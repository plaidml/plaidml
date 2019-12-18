//===- OpLibWrapperGen.cpp - MLIR op lib wrapper generator ----------------===//
//
// Copyright 2019 Intel Corporation.
//
// =============================================================================
//
// OpLibWrapperGen
//
//===----------------------------------------------------------------------===//

#pragma once

#include <vector>

#include "pmlc/dialect/op_lib/OpModel.h"

using llvm::raw_ostream;
using llvm::RecordKeeper;

namespace pmlc::dialect::op::tblgen::python {

class Emitter {
 private:
  DialectInfo info_;

 public:
  Emitter(DialectInfo info, raw_ostream& os);
  static void emitHeaders(raw_ostream& os);
  static void emitInits(raw_ostream& os);
  static void emitOps(const std::vector<OpInfo>& ops, raw_ostream& os);
};

static inline bool genWrappers(const RecordKeeper& recordKeeper, raw_ostream& os) {
  // First, grab all the data we'll ever need from the record and place it in a DialectInfo struct
  auto OpLibDialect = DialectInfo(recordKeeper);
  // Then, emit specifically for python
  auto OpLibEmitter = Emitter(OpLibDialect, os);
  return false;
}

}  // namespace pmlc::dialect::op::tblgen::python
