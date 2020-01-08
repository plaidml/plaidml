//===- OpModel.h - MLIR op lib model structs -------------------------------===//
//
// Copyright 2019 Intel Corporation.
//
// =============================================================================
//
// OpModel
//
//===----------------------------------------------------------------------===//

#pragma once

#include <map>
#include <vector>

#include "mlir/Support/STLExtras.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/Operator.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/TableGen/Record.h"

using llvm::MapVector;
using llvm::RecordKeeper;
using mlir::StringRef;
using mlir::tblgen::EnumAttr;
using mlir::tblgen::Operator;

namespace pmlc::tools::tblgen {

struct OpInfo {
  StringRef name_;
  StringRef returnType_;
  MapVector<StringRef, StringRef> attributes_;
  MapVector<StringRef, StringRef> operands_;
  explicit OpInfo(const Operator& op);
};

struct TypeInfo {
  StringRef name_;
  MapVector<StringRef, int> opts_;
  StringRef returnType_;
  explicit TypeInfo(const EnumAttr& ea);
};

struct DialectInfo {
  std::vector<OpInfo> all_ops_;
  std::vector<TypeInfo> all_types_;
  explicit DialectInfo(const RecordKeeper& recordKeeper);
};

}  // namespace pmlc::tools::tblgen
