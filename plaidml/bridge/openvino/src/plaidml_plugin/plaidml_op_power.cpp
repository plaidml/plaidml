// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_op_power.hpp"
#include "plaidml_util.hpp"

#include "plaidml/op/op.h"

void OpPower::run(const plaidml::edsl::Tensor& I, plaidml::edsl::Tensor& O) {
  auto* pow_layer = dynamic_cast<PowerLayer*>(layer_.get());
  O = pow(pow_layer->scale * I + pow_layer->offset, Tensor(pow_layer->power));
}
