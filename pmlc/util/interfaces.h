// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace pmlc {
namespace util {

using mlir::ArrayRef;
using mlir::Location;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OpInterface;
using mlir::Type;
using mlir::Value;

#include "pmlc/util/interfaces.h.inc"

} // namespace util
} // namespace pmlc
