//===- OpLibGen.cpp - MLIR op lib dialect tablegen  -----------------------===//
//
// Copyright 2019 Intel Corporation.
//
// =============================================================================
//
// Generates registrations for each of the op lib dialect's tablegen outputs.
//
//===----------------------------------------------------------------------===//

#include "mlir/TableGen/GenInfo.h"

#include "pmlc/dialect/op_lib/cpp/ImplGen.h"
#include "pmlc/dialect/op_lib/cpp/WrapperGen.h"
#include "pmlc/dialect/op_lib/python/WrapperGen.h"

using mlir::GenRegistration;

namespace pmlc::dialect::op {

static mlir::GenRegistration genOpCPPImpl("gen-op-lib-cpp-impl", "Generate Op Lib C++ EDSL impl",
                                          [](const RecordKeeper& records, raw_ostream& os) {
                                            return tblgen::cpp::genImpl(records, os);
                                          });

static mlir::GenRegistration genOpCPPWrappers("gen-op-lib-cpp-wrappers", "Generate Op Lib C++ EDSL wrappers",
                                              [](const RecordKeeper& records, raw_ostream& os) {
                                                return tblgen::cpp::genWrappers(records, os);
                                              });

static mlir::GenRegistration genOpPYWrappers("gen-op-lib-py-wrappers", "Generate Op Lib Python EDSL wrappers",
                                             [](const RecordKeeper& records, raw_ostream& os) {
                                               return tblgen::python::genWrappers(records, os);
                                             });

}  // namespace pmlc::dialect::op
