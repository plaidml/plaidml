// Copyright 2019 Intel Corporation.

#include "plaidml2/edsl/helper.h"

#include "plaidml2/core/internal.h"
#include "pmlc/conversion/tile_to_stripe/tile_to_stripe.h"
#include "pmlc/dialect/stripe/transcode.h"

namespace plaidml::edsl {

using StripeProgram = vertexai::tile::stripe::Program;
using pmlc::dialect::stripe::FromMLIR;
using pmlc::dialect::tile::LowerIntoStripe;

std::shared_ptr<StripeProgram> ConvertIntoStripe(const Program& program) {
  auto tileProgram = program.as_ptr()->program;
  auto module = LowerIntoStripe(*tileProgram->module);
  return FromMLIR(*module);
}

}  // namespace plaidml::edsl
