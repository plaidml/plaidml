// Copyright 2019, Intel Corporation

#pragma once

#include "pmlc/dialect/eltwise/types.h"

#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"

namespace mlir {

std::ostream& operator<<(std::ostream& os, ModuleOp rhs);
std::ostream& operator<<(std::ostream& os, Type rhs);
std::ostream& operator<<(std::ostream& os, const Value& rhs);
std::ostream& operator<<(std::ostream& os, const Operation& rhs);

}  // namespace mlir

namespace pmlc {
namespace dialect {
namespace eltwise {

llvm::SmallVector<int64_t, 4> ComputeShape(llvm::ArrayRef<mlir::Value*> operands);

mlir::Type ComputeResultType(llvm::ArrayRef<mlir::Value*> operands, DataType override = DataType::INVALID);

// Adjust the result types on the containing FuncOp if this op relates to an output
void UpdateFuncOpType(mlir::Operation* op);

mlir::RankedTensorType GetTensorType(mlir::Type type);
llvm::StringRef getOpName(const mlir::OperationName& name);

using UnaryCalculate = std::function<double(double)>;
mlir::Attribute constFoldUnaryOp(llvm::ArrayRef<mlir::Attribute> operands, UnaryCalculate calculate);

using BinaryCalculate = std::function<double(double, double)>;
mlir::Attribute constFoldBinaryOp(llvm::ArrayRef<mlir::Attribute> operands, BinaryCalculate calculate);

struct ConstantValueMatcher {
  double value;
  bool match(mlir::Operation* op);
};

inline ConstantValueMatcher m_Zero() { return ConstantValueMatcher{0}; }

inline ConstantValueMatcher m_One() { return ConstantValueMatcher{1}; }

}  // namespace eltwise
}  // namespace dialect
}  // namespace pmlc
