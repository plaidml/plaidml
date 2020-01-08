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

#include "pmlc/tools/pmlc-tblgen/cpp/impl_gen.h"
#include "pmlc/tools/pmlc-tblgen/cpp/wrapper_gen.h"
#include "pmlc/tools/pmlc-tblgen/python/wrapper_gen.h"

using llvm::raw_ostream;
using llvm::RecordKeeper;
using mlir::GenRegistration;

namespace pmlc::tools::tblgen {

static mlir::GenRegistration genOpCPPImpl(              //
    "gen-cpp-impl",                                     //
    "Generate Op Lib C++ EDSL impl",                    //
    [](const RecordKeeper& records, raw_ostream& os) {  //
      return cpp::genImpl(records, os);
    });

static mlir::GenRegistration genOpCPPWrappers(          //
    "gen-cpp-wrappers",                                 //
    "Generate Op Lib C++ EDSL wrappers",                //
    [](const RecordKeeper& records, raw_ostream& os) {  //
      return cpp::genWrappers(records, os);
    });

static mlir::GenRegistration genOpPYWrappers(           //
    "gen-py-wrappers",                                  //
    "Generate Op Lib Python EDSL wrappers",             //
    [](const RecordKeeper& records, raw_ostream& os) {  //
      return python::genWrappers(records, os);
    });

}  // namespace pmlc::tools::tblgen
