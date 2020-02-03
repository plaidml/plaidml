//===- OpLibWrapperGen.cpp - MLIR op lib wrapper generator ----------------===//
// Copyright 2019 Intel Corporation.
// =============================================================================
// OpLibWrapperGen
//===----------------------------------------------------------------------===
#include "pmlc/tools/pmlc-tblgen/model.h"

#include <vector>

using llvm::MapVector;
using llvm::Record;
using llvm::RecordKeeper;
using mlir::StringRef;
using mlir::tblgen::EnumAttr;
using mlir::tblgen::Operator;

namespace pmlc::tools::tblgen {

// Operators, Operands, and Attributes

static MapVector<StringRef, StringRef> getAttributes(const Operator &op) {
  MapVector<StringRef, StringRef> attributes_;
  for (auto &namedAttr : op.getAttributes()) {
    const auto &name = namedAttr.name;
    auto mlir_type = namedAttr.attr.getReturnType();
    const auto &is_enum = namedAttr.attr.isEnumAttr();
    if (is_enum) {
      mlir_type = namedAttr.attr.getAttrDefName();
    }
    attributes_.insert({name, mlir_type});
  }
  return attributes_;
}

static MapVector<StringRef, StringRef> getOperands(const Operator &op) {
  MapVector<StringRef, StringRef> operands_;
  for (int index = 0; index < op.getNumOperands(); index++) {
    auto &namedOperand = op.getOperand(index);
    const auto &name = namedOperand.name;
    const auto &mlir_type = namedOperand.constraint.getDescription();
    operands_.insert({name, mlir_type});
  }
  return operands_;
}

static StringRef getReturnType(const Operator &op) {
  StringRef type;
  int n_results = op.getNumResults();
  if (n_results > 1) {
    type = "MLIR_LIST";
  } else if (n_results == 0) {
    type = "MLIR_VOID";
  } else {
    auto &namedResult = op.getResult(0);
    type = namedResult.constraint.getDescription();
  }
  return type;
}

// OpInfo initializer
OpInfo::OpInfo(const Operator &op) {
  name_ = op.getCppClassName();
  returnType_ = getReturnType(op);
  attributes_ = getAttributes(op);
  operands_ = getOperands(op);
}

// Record accesses for ops
std::vector<OpInfo> getOpRecords(const RecordKeeper &recordKeeper) {
  std::vector<OpInfo> all_ops;
  auto pmlOpRecords = recordKeeper.getAllDerivedDefinitions("PML_Op");
  for (auto *record : pmlOpRecords) {
    all_ops.push_back(OpInfo(Operator(*record)));
  }
  return all_ops;
}

// Types
// Each custom type has its own TypeInfo initializer, and needs to be included
// in getTypeRecords so that the Record is obtained.

// TypeInfo initializer for a single EnumAttr input.
TypeInfo::TypeInfo(const EnumAttr &ea) {
  name_ = ea.getEnumClassName();
  returnType_ = ea.getUnderlyingType();
  for (auto eacase : ea.getAllCases()) {
    opts_.insert({eacase.getSymbol(), eacase.getValue()});
  }
}

// Record accesses for types
// As we add more types, we can add them to TypeInfo through here
std::vector<TypeInfo> getTypeRecords(const RecordKeeper &recordKeeper) {
  std::vector<TypeInfo> all_types;
  auto enumTypeRecords = recordKeeper.getAllDerivedDefinitions("EnumAttrInfo");
  for (auto *record : enumTypeRecords) {
    all_types.push_back(TypeInfo(EnumAttr(*record)));
  }
  return all_types;
}

// Dialects

DialectInfo::DialectInfo(const RecordKeeper &recordKeeper)
    : all_ops_(getOpRecords(recordKeeper)),
      all_types_(getTypeRecords(recordKeeper)) {}

} // namespace pmlc::tools::tblgen
