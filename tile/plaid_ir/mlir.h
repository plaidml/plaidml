// Copyright 2019, Intel Corporation

#pragma once

#include <tuple>
#include <vector>

// Include things we need mostly everywhere anyway

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#include "mlir/Support/STLExtras.h"

namespace vertexai {
namespace tile {
namespace plaid_ir {

// We load some things from the LLVM / MLIR namespace into our own namespace
// and add some wrappers for various classes to make life easier.  In general,
// I'd like to do less of this, but unfortunately .td codegen process is
// hardcoded to emit things like 'StringRef' without any namespace prefix, and I
// need to at least include ops.h without a 'using namespace'

using mlir::APInt;
using mlir::ArrayAttr;
using mlir::ArrayRef;
using mlir::Attribute;
using mlir::Block;
using mlir::Builder;
using mlir::IntegerAttr;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::NamedAttribute;
using mlir::NamedAttributeList;
using mlir::Op;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OperationState;
using mlir::Region;
using mlir::StringAttr;
using mlir::StringRef;
using mlir::Type;
using mlir::Value;

}  // namespace plaid_ir
}  // namespace tile
}  // namespace vertexai

// Helpers for LLVM hashing, implemented in the namespace of the things we are
// hashing to allow ADL to work.

namespace std {

template <typename T>
inline llvm::hash_code hash_value(const std::vector<T>& vec) {
  return llvm::hash_combine_range(vec.begin(), vec.end());
}

template <typename Tuple, typename std::size_t... I>
inline llvm::hash_code tuple_hash_value_impl(const Tuple& tup, std::index_sequence<I...> idxs) {
  return llvm::hash_combine(std::get<I>(tup)...);
}

template <typename... Args>
inline llvm::hash_code hash_value(const std::tuple<Args...>& tup) {
  return tuple_hash_value_impl(tup, std::index_sequence_for<Args...>());
}

}  // namespace std

namespace mlir {

inline llvm::hash_code hash_value(const NamedAttributeList& attrs) { return hash_value(attrs.getDictionary()); }

}  // namespace mlir
