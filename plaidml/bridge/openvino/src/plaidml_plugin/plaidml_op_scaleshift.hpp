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
PLAIDML_LAYER(OpScaleShift, <Tensor(const Tensor&, const Tensor&, const Tensor&)>, "scaleshift") {
  void run(const Tensor&, const Tensor&, const Tensor&, Tensor&);

 private:
  void LoadWeights(State * state) override;

  ScaleShiftLayer* scaleshift_layer_;
};

}  // namespace PlaidMLPlugin
