//===- WrapperGen.h - MLIR op lib dialect wrapper generator for Python ----===//
//
// Copyright 2019 Intel Corporation.
//
// =============================================================================
//
// WrapperGen in Python generates a file that wraps the Python op lib with a
// fluent interface.
//
//===----------------------------------------------------------------------===//

#include <vector>

#include "pmlc/dialect/op_lib/python/WrapperGen.h"
#include "pmlc/dialect/op_lib/python/utils.h"

using llvm::raw_ostream;
using llvm::RecordKeeper;

namespace pmlc::dialect::op::tblgen::python {

Emitter::Emitter(DialectInfo info, raw_ostream& os) : info_(info) {
  emitHeaders(os);
  emitInits(os);
  emitOps(info.all_ops_, os);
}
void Emitter::emitHeaders(raw_ostream& os) { os << fileCommentHeader << includeHeader; }
void Emitter::emitInits(raw_ostream& os) { os << initFunction << ffiFunction; }
void Emitter::emitOps(const std::vector<OpInfo>& ops, raw_ostream& os) {
  for (auto op : ops) {
    os << commentHeader << "## " << op.name_ << commentHeader;
    // OpEmitter(op, os);
  }
}

}  // namespace pmlc::dialect::op::tblgen::python
