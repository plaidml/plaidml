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
#include "pmlc/dialect/op_lib/cpp/WrapperGen.h"
#include "pmlc/dialect/op_lib/python/WrapperGen.h"

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

static mlir::GenRegistration genOpCPPImpl("gen-op-lib-cpp-impl", "Generate Op Lib C++ EDSL impl",
                                          [](const RecordKeeper& records, raw_ostream& os) {
                                            return tblgen::cpp::genImpl(records, os);
                                          });

static mlir::GenRegistration genOpCPPWrappers("gen-op-lib-cpp-wrappers", "Generate Op Lib C++ EDSL wrappers",
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
