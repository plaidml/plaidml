// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "plaidml/op/op.h"

#include "plaidml_op_norm.hpp"
#include "plaidml_util.hpp"

void OpNorm::run(const plaidml::edsl::Tensor& I, plaidml::edsl::Tensor& O) {
  auto norm_layer_ = dynamic_cast<NormLayer*>(layer_.get());
  IE_ASSERT(norm_layer_->outData.size() == 1);
  IE_ASSERT(norm_layer_->_isAcrossMaps == true);
  // acrossMaps = true  | average over channels
  //              false | average within channel (over 2D window)
  const auto& out_dims = util::to_plaidml(norm_layer_->outData.front()->getTensorDesc().getDims());
  IE_ASSERT(out_dims.size() == 4);
  float div_size = 1.0 / norm_layer_->_size;
  TensorIndex n, c, h, w, s;
  O = TensorOutput(out_dims[0], out_dims[2], out_dims[3], out_dims[1]);
  O(n, h, w, c) += I(n, h, w, c + s - norm_layer_->_size / 2) * I(n, h, w, c + s - norm_layer_->_size / 2);
  O.add_constraint(s < TensorDim(norm_layer_->_size));
  O = I * plaidml::edsl::pow(Tensor(norm_layer_->_k) + norm_layer_->_alpha * div_size * O, Tensor(-norm_layer_->_beta));
}
