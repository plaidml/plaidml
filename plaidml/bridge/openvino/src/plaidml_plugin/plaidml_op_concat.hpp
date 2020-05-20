// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <utility>
#include <vector>

#include <ie_layers.h>

#include "plaidml/edsl/edsl.h"

#include "plaidml_op.hpp"
#include "plaidml_state.hpp"

using namespace InferenceEngine;

namespace PlaidMLPlugin {

using namespace plaidml::edsl;
PLAIDML_LAYER(OpConcat, <Tensor(const std::vector<Tensor>&)>, "concat") {
  void run(const std::vector<Tensor>&, Tensor&);
  void PackInputs(State * state) override;
};

}  // namespace PlaidMLPlugin
