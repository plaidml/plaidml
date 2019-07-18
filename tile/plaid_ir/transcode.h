// Copyright 2019, Intel Corporation

#pragma once

#include "tile/plaid_ir/ops.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace plaid_ir {

mlir::FuncOp StripeToPlaidIR(MLIRContext* ctx, const stripe::Program& prog);
stripe::Program PlaidIRToStripe(const mlir::FuncOp& func);

}  // namespace plaid_ir
}  // namespace tile
}  // namespace vertexai
