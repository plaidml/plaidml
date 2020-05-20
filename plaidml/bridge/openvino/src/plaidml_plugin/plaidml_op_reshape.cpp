// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "plaidml/op/op.h"

#include "plaidml_op_reshape.hpp"
#include "plaidml_util.hpp"

void OpReshape::run(const plaidml::edsl::Tensor& I, plaidml::edsl::Tensor& O) {
  const auto& in_desc = layer_->outData.front()->getTensorDesc();
  const auto& out_dims = util::to_plaidml(in_desc.getDims());
  const auto& in_layout = layer_->insData.front().lock()->getTensorDesc().getLayout();
  const auto& out_layout = in_desc.getLayout();

  /* FIXME:
   * We put NCHW -> NHWC conversion at the beginning of the network
   * So we can't do the reshape correctly we need to do:
   * (NHWC->NCHW)conversion => reshape => (NCHW->NHWC)conversion
   */
  if (in_layout == NCHW) {
    O = plaidml::op::transpose(I, Value{{Value{0}, Value{3}, Value{1}, Value{2}}});
    O = plaidml::edsl::reshape(O, out_dims);
  } else {
    O = plaidml::edsl::reshape(I, out_dims);
  }

  if (out_layout == NCHW) {
    O = plaidml::op::transpose(O, Value{{Value{0}, Value{2}, Value{3}, Value{1}}});
  }
}
