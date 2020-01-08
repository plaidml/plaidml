// Copyright 2019, Intel Corporation

#pragma once

#include <set>
#include <string>
#include <utility>

#include "mlir/IR/Function.h"

#include "pmlc/dialect/eltwise/ir/ops.h"
#include "pmlc/dialect/stripe/dialect.h"
#include "pmlc/dialect/stripe/mlir.h"
#include "pmlc/dialect/stripe/ops.h"

namespace pmlc {
namespace dialect {
namespace stripe {

/// Wraps function body with a ParallelForOp to represent Stripe's 'main' block.
void createMainParallelFor(mlir::FuncOp funcOp);

// Add an attribute in a DictionaryAttr
DictionaryAttr addAttrInDictionary(DictionaryAttr old_dict, OpBuilder builder, NamedAttribute elem);

// Add an attribute in an ArrayAttr
DictionaryAttr addAttrInArray(DictionaryAttr old_dict, OpBuilder builder, Attribute elem);

// Replace the n-th element in a DictionaryAttr
DictionaryAttr replaceAttrInDictionary(DictionaryAttr old_dict, OpBuilder builder, unsigned n, NamedAttribute elem);

// Replace the n-th element in an ArrayAttr
ArrayAttr replaceAttrInArray(ArrayAttr old_array, OpBuilder builder, unsigned n, Attribute elem);

// Check if op has attr
bool hasAttr(mlir::Operation* op, StringRef attr);

// Check if op has attrs
bool hasAttrs(mlir::Operation* op, const std::set<std::string>& attrs);

// Add a unit attribute for op
void setOpAttrUnit(mlir::Operation* op, mlir::OpBuilder builder, StringRef attr);

// Add a unit attribute for target_idx
void setIdxAttrUnit(ParallelForOp op, StringRef target_idx, StringRef attr);

// Get the index range
int64_t idxRange(mlir::BlockArgument idx);

// Get the index name
StringRef idxName(mlir::BlockArgument idx);

// Get the value name. value must be a tensor.
StringRef tensorName(Value tensor);

// Get the element type of tensor
DataType tensorElementType(Value tensor);

// Get a single index from ParallelForOp
std::pair<StringRef, unsigned> getSingleIndex(ParallelForOp op, unsigned n);

// Get all index from ParallelForOp
void getAllIndex(ParallelForOp op, llvm::SmallVectorImpl<std::pair<StringRef, unsigned>>* into);

// Get the base type for a tensor
TensorType baseType(Value value);

// Get all index in tensor "value", whose strides are 1
void strideOneIdxs(mlir::Value value, llvm::SmallVectorImpl<mlir::BlockArgument>* into);

// Build the initial value for the aggregate type
eltwise::ScalarConstantOp initialValue(  //
    OpBuilder* builder,                  //
    DataType type,                       //
    AggregationKind agg,                 //
    StringRef var_name);

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
