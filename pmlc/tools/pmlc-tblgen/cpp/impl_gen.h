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

#include "llvm/TableGen/Record.h"

namespace pmlc::tools::tblgen::cpp {

bool genImpl(const llvm::RecordKeeper& recordKeeper, llvm::raw_ostream& os);

}  // namespace pmlc::tools::tblgen::cpp
