//===- utils.cc - MLIR op lib dialect tblgen utils for Python -------------===//
//
// Copyright 2019 Intel Corporation.
//
// =============================================================================
//
// Tablegen utilities for the Python outputs of the op lib dialect. This
// encompasses things such as headers, includes, and type mappings.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <map>

#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

using mlir::StringRef;

namespace pmlc::tools::tblgen::python {

static inline const char* const fileCommentHeader = R"(## Copyright 2019 Intel Corporation.
##===- TableGen'erated file ----------------------------------*- Python -*-===##
##                                                                            ##
## Op Lib Python EDSL Wrapper                                                 ##
##                                                                            ##
## Automatically generated file, do not edit!                                 ##
##                                                                            ##
##===----------------------------------------------------------------------===##

)";

static inline const char* const includeHeader = R"(
import logging

import six

import plaidml2.edsl as edsl
from plaidml2.ffi import ffi, ffi_call, lib

)";

static inline const char* const commentHeader = R"(
##===----------------------------------------------------------------------===##
)";

static inline const char* const ffiFunction = R"(
def op(op_name, args):
    value = edsl.Value(args)
    return edsl.Value(ffi_call(lib.plaidml_op_make, op_name.encode(), value.as_ptr()))

)";

static inline const char* const initFunction = R"(
logger = logging.getLogger(__name__)


def __init():
    ffi_call(lib.plaidml_op_init)


ffi.init_once(__init, 'plaidml_op_init')

)";

}  // namespace pmlc::tools::tblgen::python
