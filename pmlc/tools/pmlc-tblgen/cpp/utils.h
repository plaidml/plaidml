//===- utils.cc - MLIR op lib dialect tblgen utils for C++ ----------------===//
//
// Copyright 2019 Intel Corporation.
//
// =============================================================================
//
// Tablegen utilities for the C++ outputs of the op lib dialect. This
// encompasses things such as headers, includes, and type mappings.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <map>

#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/StringExtras.h"

using mlir::StringRef;

namespace pmlc::tools::tblgen::cpp {

static const char* const fileCommentHeader = R"( // Copyright 2019 Intel Corporation.
/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Lib C++ EDSL Wrapper                                                    *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

)";

static const char* const includeHeader = R"(
#pragma once

#include <string>
#include <vector>

#include "plaidml/edsl/edsl.h"
#include "plaidml/op/ffi.h"

)";

static const char* const commentHeader = R"(
//===----------------------------------------------------------------------===//
// {0} {1}
//===----------------------------------------------------------------------===//

)";

static const char* const ffiFunction = R"(
namespace details {

inline edsl::Value op(const std::string& name, const edsl::Value& args) {
  return edsl::Value(ffi::call<plaidml_value*>(plaidml_op_make, name.c_str(), args.as_ptr()));
}

} // namespace details

)";

static const char* const initFunction = R"(
inline void init() {  //
  plaidml::init();
  plaidml::edsl::init();
  ffi::call_void(plaidml_op_init);
}

)";

// Order doesn't matter here, but lookup efficiency does, so keep as a std::map.
static const std::map<StringRef, StringRef> typeLookupTable = {
    {"APInt", "int"},
    {"ArrayAttr", "std::vector<int>"},
    {"bool", "bool"},
    {"MLIR_LIST", "edsl::Value&"},
    {"MLIR_VOID", "void"},
    {"PML_AutogroupModeAttr", "AutogroupMode"},
    {"PML_AutopadModeAttr", "AutopadMode"},
    {"PML_DerivModeAttr", "DerivMode"},
    {"PML_GroupLayoutAttr", "GroupLayout"},
    {"PML_TensorLayoutAttr", "TensorLayout"},
    {"StringRef", "std::string"},
    {"tensor of any type values", "edsl::Tensor"},
    {"uint32_t", "int64_t"},
};

static inline StringRef convertType(const StringRef type) {
  StringRef edslType = "unrecognized";
  auto found = typeLookupTable.find(type);
  if (found != typeLookupTable.end()) {
    edslType = found->second;
  }
  return edslType;
}

}  // namespace pmlc::tools::tblgen::cpp
