// Copyright 2019, Intel Corporation

#pragma once

#include "pmlc/dialect/mir/mlir.h"
#include "tile/stripe/stripe.h"

namespace pmlc {
namespace dialect {
namespace mir {

namespace stripe = vertexai::tile::stripe;

mlir::FuncOp ToMir(MLIRContext* ctx, const stripe::Program& prog);
stripe::Program ToStripe(mlir::FuncOp func);

}  // namespace mir
}  // namespace dialect
}  // namespace pmlc
