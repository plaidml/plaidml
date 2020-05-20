// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_op_relu.hpp"
#include "plaidml_util.hpp"

#include "plaidml/op/op.h"

void OpReLU::run(const plaidml::edsl::Tensor& I, plaidml::edsl::Tensor& O) {
  auto* layer = dynamic_cast<ReLULayer*>(layer_.get());
  O = plaidml::op::relu(I).alpha(Tensor(layer->negative_slope));
}
