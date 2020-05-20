// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_op_clamp.hpp"
#include "plaidml_util.hpp"

#include "plaidml/op/op.h"

void OpClamp::run(const plaidml::edsl::Tensor& I, plaidml::edsl::Tensor& O) {
  auto clamp = dynamic_cast<ClampLayer*>(layer_.get());

  O = plaidml::edsl::select(I > clamp->max_value, Tensor(clamp->max_value), I);
  O = plaidml::edsl::select(O < clamp->min_value, Tensor(clamp->min_value), O);
}
