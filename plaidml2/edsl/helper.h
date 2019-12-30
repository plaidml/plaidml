// Copyright 2019 Intel Corporation.

#pragma once

#include <memory>

#include "plaidml2/edsl/edsl.h"
#include "tile/stripe/stripe.h"

namespace plaidml::edsl {

std::shared_ptr<vertexai::tile::stripe::Program> ConvertIntoStripe(const Program& program);

}  // namespace plaidml::edsl
