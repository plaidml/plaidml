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

/* Convolution works with 2 or 3 input tensors
 * so we don't use macro PLAIDML_LAYER with hardcoded API
 */
struct OpConvolution : public Op {
  void run(const plaidml::edsl::Tensor& I, const plaidml::edsl::Tensor& K, const plaidml::edsl::Tensor& B,
           plaidml::edsl::Tensor& O);

  void run(const plaidml::edsl::Tensor& I, const plaidml::edsl::Tensor& K, plaidml::edsl::Tensor& O);

  void LoadWeights(State* state) override;

  void Execute() override;

  ConvolutionLayer* conv_layer_;
};

}  // namespace PlaidMLPlugin
