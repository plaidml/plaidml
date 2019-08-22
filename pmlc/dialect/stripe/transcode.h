// Copyright 2019, Intel Corporation

#pragma once

#include "pmlc/dialect/stripe/mlir.h"
#include "tile/stripe/stripe.h"

namespace pmlc {
namespace dialect {
namespace stripe {

namespace stripe = vertexai::tile::stripe;

mlir::FuncOp ToStripeMLIR(MLIRContext* ctx, const stripe::Program& prog);
stripe::Program ToStripe(mlir::FuncOp func);

}  // namespace stripe
}  // namespace dialect
}  // namespace pmlc
