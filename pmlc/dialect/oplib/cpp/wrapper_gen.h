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

#include "llvm/TableGen/Record.h"

namespace pmlc::dialect::oplib::cpp {

bool genWrappers(const llvm::RecordKeeper& recordKeeper, llvm::raw_ostream& os);

}  // namespace pmlc::dialect::oplib::cpp
