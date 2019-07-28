// Copyright 2019, Intel Corporation

#pragma once

#include <map>
#include <string>

#include "mlir/IR/Builders.h"

#include "tile/stripe/stripe.h"

#include "pmlc/dialect/scalar/ops.h"

namespace pmlc {
namespace dialect {
namespace scalar {

namespace stripe = vertexai::tile::stripe;

struct SymbolTable {
  std::map<std::string, mlir::Value*> refs;
  std::map<std::string, mlir::Value*> idxs;
  std::map<std::string, mlir::Value*> scalars;
};

void IntrinsicToScalarOp(mlir::OpBuilder* builder, SymbolTable* locals, const stripe::Intrinsic& intrinsic);

ScalarType ToPlaidIR(mlir::MLIRContext* ctx, vertexai::tile::DataType dtype);

}  // namespace scalar
}  // namespace dialect
}  // namespace pmlc
