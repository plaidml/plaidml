// Copyright 2019, Intel Corporation

#pragma once

#include "pmlc/dialect/mir/ops.h"
#include "tile/stripe/stripe.h"

namespace pmlc {
namespace dialect {
namespace mir {

namespace stripe = vertexai::tile::stripe;

mlir::FuncOp StripeToPlaidIR(MLIRContext* ctx, const stripe::Program& prog);
stripe::Program PlaidIRToStripe(mlir::FuncOp func);

}  // namespace mir
}  // namespace dialect
}  // namespace pmlc
