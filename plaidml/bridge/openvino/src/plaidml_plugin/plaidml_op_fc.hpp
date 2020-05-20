// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
/*
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
PLAIDML_LAYER(OpFc, <Tensor(const Tensor&, const Tensor&, const Tensor&)>, "fullyconnected") {
  void run(const Tensor&, const Tensor&, const Tensor&, Tensor&);

 private:
  void LoadWeights(State * state) override;

  FullyConnectedLayer* fc_layer_;
};

}  // namespace PlaidMLPlugin
*/