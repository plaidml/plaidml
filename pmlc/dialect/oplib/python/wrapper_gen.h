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

#include "llvm/TableGen/Record.h"

namespace pmlc::dialect::oplib::python {

bool genWrappers(const llvm::RecordKeeper& recordKeeper, llvm::raw_ostream& os);

}  // namespace pmlc::dialect::oplib::python
