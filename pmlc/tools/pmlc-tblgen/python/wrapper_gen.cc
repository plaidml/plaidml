//===- WrapperGen.h - MLIR op lib dialect wrapper generator for Python ----===//
// Copyright 2019 Intel Corporation.
// =============================================================================
// WrapperGen in Python generates a file that wraps the Python op lib with a
// fluent interface.
//===----------------------------------------------------------------------===
#include "pmlc/tools/pmlc-tblgen/python/wrapper_gen.h"

#include <vector>

#include "pmlc/tools/pmlc-tblgen/model.h"
#include "pmlc/tools/pmlc-tblgen/python/utils.h"

using llvm::raw_ostream;
using llvm::RecordKeeper;

namespace pmlc::tools::tblgen::python {

class Emitter {
private:
  DialectInfo info_;

public:
  Emitter(DialectInfo info, raw_ostream &os);
  static void emitHeaders(raw_ostream &os);
  static void emitInits(raw_ostream &os);
  static void emitOps(const std::vector<OpInfo> &ops, raw_ostream &os);
};

Emitter::Emitter(DialectInfo info, raw_ostream &os) : info_(info) {
  emitHeaders(os);
  emitInits(os);
  emitOps(info.all_ops_, os);
}
void Emitter::emitHeaders(raw_ostream &os) {
  os << fileCommentHeader << includeHeader;
}
void Emitter::emitInits(raw_ostream &os) { os << initFunction << ffiFunction; }
void Emitter::emitOps(const std::vector<OpInfo> &ops, raw_ostream &os) {
  for (auto op : ops) {
    os << commentHeader << "## " << op.name_ << commentHeader;
    // OpEmitter(op, os);
  }
}

bool genWrappers(const RecordKeeper &recordKeeper, raw_ostream &os) {
  // First, grab all the data we'll ever need from the record and place it in a
  // DialectInfo struct
  auto OpLibDialect = DialectInfo(recordKeeper);
  // Then, emit specifically for python
  auto OpLibEmitter = Emitter(OpLibDialect, os);
  return false;
}

} // namespace pmlc::tools::tblgen::python
