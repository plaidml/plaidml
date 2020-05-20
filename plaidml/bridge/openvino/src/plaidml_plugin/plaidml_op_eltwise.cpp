// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
/*
#include "plaidml_op_eltwise.hpp"
#include "plaidml_util.hpp"

#include "plaidml/op/op.h"

void OpEltwise::run(const plaidml::edsl::Tensor& I1, const plaidml::edsl::Tensor& I2, plaidml::edsl::Tensor& O) {
  auto* eltwise = dynamic_cast<EltwiseLayer*>(layer_.get());

  // FIXME: Support more eltwise operations
  switch (eltwise->_operation) {
    case EltwiseLayer::eOperation::Sum:
      O = I1 + I2;
      break;
    case EltwiseLayer::eOperation::Sub:
      O = I1 - I2;
      break;
    case EltwiseLayer::eOperation::Prod:
      O = I1 * I2;
      break;
    case EltwiseLayer::eOperation::Div:
      O = I1 / I2;
      break;
    default:
      THROW_IE_EXCEPTION << "Unsupported operation type for Eltwise";
  }
}
*/