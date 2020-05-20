// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "plaidml/op/op.h"
#include "plaidml_op_crop.hpp"
#include "plaidml_util.hpp"
#include <functional>

void OpCrop::run(const plaidml::edsl::Tensor& I, plaidml::edsl::Tensor& O) {
  auto crop_layer_ = dynamic_cast<CropLayer*>(layer_.get());
  const auto& out_dims = util::to_plaidml(crop_layer_->outData.front()->getTensorDesc().getDims());
  const auto& in_dims = util::to_plaidml(crop_layer_->insData.front().lock()->getDims());
  const auto& in_axis = crop_layer_->axis;
  const auto& in_offset = crop_layer_->offset;
  const auto& in_layout = crop_layer_->insData.front().lock()->getTensorDesc().getLayout();
  const auto& out_layout = crop_layer_->outData.front()->getTensorDesc().getLayout();

  IE_ASSERT((in_dims.size() == 4 || in_dims.size() == 2) && "Crop layer supports only 2D and 4D input tensors");
  // Data to vector, because
  // data from IR can be a vector with various sizes.
  // We put offset information in its place
  std::vector<int> offset(in_dims.size(), 0);
  for (int i = 0; i < in_axis.size(); ++i) {
    offset[in_axis[i]] = in_offset[i];
  }
  Tensor FLATTENED;
  TensorIndex n, c, h, w;
  std::vector<int64_t> dims_after_reshape;
  dims_after_reshape.emplace_back(std::accumulate(in_dims.begin(), in_dims.end(), 1, std::multiplies<size_t>()));
  if (in_layout == NCHW) {
    O = TensorOutput(out_dims[0], out_dims[1], out_dims[2], out_dims[3]);
    FLATTENED = plaidml::op::transpose(I, Value{{Value{0}, Value{3}, Value{1}, Value{2}}});
    // Data shift first, then crop.
    // shift the flattened tensor easier, reshape from [N C H W] to line [N*C*H*W].
    FLATTENED = plaidml::edsl::reshape(FLATTENED, dims_after_reshape);
    O(n, c, h, w) =
        FLATTENED((n + offset[0]) * in_dims[1] * in_dims[2] * in_dims[3] + (c + offset[1]) * in_dims[2] * in_dims[3] +
                  (h + offset[2]) * in_dims[3] + (w + offset[3]));
  } else if (in_layout == NHWC) {
    O = TensorOutput(out_dims[0], out_dims[2], out_dims[3], out_dims[1]);
    FLATTENED = plaidml::edsl::reshape(I, dims_after_reshape);
    O(n, h, w, c) =
        FLATTENED((n + offset[0]) * in_dims[2] * in_dims[3] * in_dims[1] + (c + offset[1]) * in_dims[2] * in_dims[3] +
                  (h + offset[2]) * in_dims[3] + (w + offset[3]));
  } else {
    O = TensorOutput(out_dims[0], out_dims[1]);
    FLATTENED = plaidml::edsl::reshape(I, dims_after_reshape);
    O(h, w) = FLATTENED((h + offset[0]) * in_dims[1] + (w + offset[1]));
  }
  if (out_layout == NCHW) {
    O = plaidml::op::transpose(O, Value{{Value{0}, Value{2}, Value{3}, Value{1}}});
  }
}
