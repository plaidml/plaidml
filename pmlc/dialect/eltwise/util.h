// Copyright 2019, Intel Corporation

#pragma once

#include "pmlc/dialect/eltwise/types.h"

#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Value.h"

namespace pmlc {
namespace dialect {
namespace eltwise {

llvm::SmallVector<int64_t, 4> ComputeShape(llvm::ArrayRef<mlir::Value> operands);

mlir::Type ComputeResultType(mlir::ValueRange operands, DataType override = DataType::INVALID);

mlir::RankedTensorType getRankedTensorType(mlir::Type type);

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
