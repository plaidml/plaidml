//===- OpLibWrapperGen.cpp - MLIR op lib wrapper generator ----------------===//
//
// Copyright 2019 Intel Corporation.
//
// =============================================================================
//
// OpLibWrapperGen
//
//===----------------------------------------------------------------------===//

#include <vector>

#include "pmlc/dialect/op_lib/python/WrapperGen.h"
#include "pmlc/dialect/op_lib/python/utils.h"

using llvm::MapVector;
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

namespace tblgen {

using namespace pmlc::dialect::op;  // NOLINT [build/namespaces]

namespace python {

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

}  // namespace python

}  // namespace tblgen

}  // namespace op
}  // namespace dialect
}  // namespace pmlc
