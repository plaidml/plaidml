// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

#include <ie_layers.h>

#include "plaidml/edsl/edsl.h"

#include "plaidml_op.hpp"
#include "plaidml_state.hpp"

using namespace InferenceEngine;

namespace PlaidMLPlugin {

using namespace plaidml::edsl;

PLAIDML_LAYER(OpPower, <Tensor(const Tensor&)>, "power") {
  void run(const plaidml::edsl::Tensor& I, plaidml::edsl::Tensor& O);
};

}  // namespace PlaidMLPlugin
