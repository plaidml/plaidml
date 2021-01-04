// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plaidml_ops.hpp"
#include "plaidml_util.hpp"
#include "llvm/Support/FormatVariadic.h"

#include "ngraph/opsets/opset.hpp"
#include "ngraph/opsets/opset1.hpp"

#include "plaidml/op/op.h"

using namespace plaidml;          // NOLINT[build/namespaces]
using namespace InferenceEngine;  // NOLINT[build/namespaces]
using namespace edsl;
using namespace plaidml::op;

namespace PlaidMLPlugin {
/*
size_t compute_conv_rank_validating_strides(std::vector<int64_t>* strides, const Tensor& I,
                                            TensorLayout input_layout) {
  size_t spatial_rank = strides->size();
  if (spatial_rank == 0) {
    spatial_rank = I.rank() - 2;
    for (size_t i = 0; i < spatial_rank; i++) {
      strides->push_back(1);
    }
  }
  return spatial_rank;
}
*/
std::pair<TensorDim, TensorDim> compute_padding_and_output_size(  //
    const TensorDim& input_size,                                  //
    const TensorDim& filter_size,                                 //
    int64_t stride,                                               //
    AutoPadMode autopad_mode,                                     //
    int64_t pad_lo,                                               //
    int64_t pad_hi,                                               //
    int64_t dilation,                                             //
    int64_t data_dilation,                                        //
    bool use_ceil_for_output_shape) {
  // Effective input and filter sizes are the sizes after dilations are
  // accounted for. So a 4x3 filter dilated by (3, 2) has an effective filter
  // size of 11 and 5 for its 2 spatial dims

  auto I_eff = (data_dilation * (input_size - 1)) + 1;  // Effective Input Size
  auto F_eff = (dilation * (filter_size - 1)) + 1;      // Effective Filter Size
  int64_t ceil_term =
      use_ceil_for_output_shape ? stride - 1 : 0;  // TODO: Will need to confirm that this is the intended behavior
  if (autopad_mode == AutoPadMode::EXPLICIT) {
    TensorDim pad_before(pad_lo);
    TensorDim output_size((I_eff + pad_lo + pad_hi - F_eff + stride + ceil_term) / stride);
    return std::pair<TensorDim, TensorDim>(pad_before, output_size);
  }
  if (autopad_mode == AutoPadMode::VALID) {
    TensorDim pad_before(0);
    TensorDim output_size((I_eff - F_eff + stride + ceil_term) / stride);
    return std::pair<TensorDim, TensorDim>(pad_before, output_size);
  }
  if (autopad_mode == AutoPadMode::SAME_LOWER || autopad_mode == AutoPadMode::SAME_UPPER) {
    TensorDim output_size((I_eff + stride - 1 + ceil_term) / stride);
    int64_t lower_term = (autopad_mode == AutoPadMode::SAME_LOWER) ? 1 : 0;
    TensorDim pad_before((max(0, (output_size - 1) * stride + F_eff - I_eff) + lower_term) / 2);
    return std::pair<TensorDim, TensorDim>(pad_before, output_size);
  }
  // throw std::runtime_error(llvm::formatv("Unexpected autopadding mode: {0}", to_string(autopad_mode)));
  THROW_IE_EXCEPTION << "Unexpected autopadding mode.";
}

void registerDeformableConvolution() {
  registerOp("DeformableConvolution", [](const Context& ctx) {
    auto* layer = ngraph::as_type<ngraph::opset1::DeformableConvolution>(ctx.layer);
    IE_ASSERT(ctx.operands.size() == 3);
    auto I = ctx.operands.at(0);
    auto OFF = ctx.operands.at(1);
    auto F = ctx.operands.at(2);
    auto name = "DeformableConvolution";
    // Get the strides
    std::vector<int> strides;
    for (auto stride : layer->get_strides()) {
      strides.push_back(stride);
    }
    // Get the dilations
    std::vector<int> dilations;
    for (auto dilation : layer->get_dilations()) {
      dilations.push_back(dilation);
    }
    auto autopad_mode = to_plaidml(layer->get_auto_pad());

    Tensor O;
    Contraction OC;
    auto spatial_rank = strides.size();

    // Prepare dimension and index variables
    TensorDim N, CI, CO;
    // The channel dimensions as used by the filters, adjusted for group layout
    TensorDim F_CI, F_CO;
    TensorDim OFF_C;
    TensorIndex n("n");
    TensorIndex ci("ci");
    TensorIndex co("co");
    TensorIndex g("g");
    TensorIndex off_c("off_c");
    TensorIndex dg("dg");
    std::vector<TensorDim> OFF_spatial_dims(spatial_rank);

    // The spatial dimensions of I
    std::vector<TensorDim> I_spatial_dims(spatial_rank);
    // The spatial indexes of I
    std::vector<TensorIndex> x;
    for (size_t i = 0; i < spatial_rank; ++i) {
      x.emplace_back(TensorIndex(llvm::formatv("x{0}", i)));
    }
    // The spatial dimensions of O; nearly unused
    std::vector<TensorDim> O_spatial_dims(spatial_rank);
    // The spatial dimensions of F
    std::vector<TensorDim> F_spatial_dims(spatial_rank);
    // The spatial indexs of F
    std::vector<TensorIndex> k;
    for (size_t i = 0; i < spatial_rank; ++i) {
      k.emplace_back(TensorIndex(llvm::formatv("k{0}", i)));
    }
    std::vector<TensorDim> I_dims;
    std::vector<TensorIndex> I_idxs;
    std::vector<TensorDim> OFF_dims;
    // std::vector<TensorIndex> OFF_idxs;
    std::vector<std::vector<TensorIndex>> OFF_idxs;
    std::vector<TensorDim> F_dims;
    std::vector<TensorIndex> F_idxs;
    std::vector<TensorDim> O_dims;
    std::vector<TensorIndex> O_idxs;
    auto group = layer->get_group();
    auto deformable_group = layer->get_deformable_group();
    TensorDim G(group);
    TensorDim DG(deformable_group);
    // The input data dims
    I_dims.push_back(N);
    I_dims.push_back(CI);
    for (size_t i = 0; i < spatial_rank; ++i) {
      I_dims.push_back(I_spatial_dims[i]);
    }
    I.bind_dims(I_dims);
    // The size of filter
    // size_t f_size = 1;
    TensorDim f_size(1);
    for (size_t i = 0; i < spatial_rank; ++i) {
      f_size = f_size * F_spatial_dims[i];
    }
    // The filter dims
    F_dims.push_back(G);
    F_dims.push_back(F_CO);
    F_dims.push_back(F_CI);
    for (size_t i = 0; i < spatial_rank; ++i) {
      F_dims.push_back(F_spatial_dims[i]);
    }
    F.bind_dims(F_dims);
    // Compute manual_padding
    std::vector<int> manual_padding;
    if (autopad_mode == plaidml::op::AutoPadMode::EXPLICIT) {
      for (auto pad : layer->get_pads_begin()) {
        manual_padding.push_back(pad);
      }
      for (auto pad : layer->get_pads_end()) {
        manual_padding.push_back(pad);
      }
    } else {
      while (manual_padding.size() < 2 * spatial_rank) {
        manual_padding.push_back(0);
      }
    }
    // Compute the output_size
    std::vector<TensorDim> pad_before;
    O_spatial_dims.clear();
    for (size_t i = 0; i < spatial_rank; ++i) {
      TensorDim local_pad_before;
      TensorDim local_output_size;
      TensorDim local_input_size;
      TensorDim local_filter_size;
      local_input_size = I_spatial_dims[i];
      local_filter_size = F_spatial_dims[i];
      std::tie(local_pad_before, local_output_size) =
          compute_padding_and_output_size(local_input_size, local_filter_size, strides[i], autopad_mode,
                                          manual_padding[i], manual_padding[i + spatial_rank], dilations[i], 1, false);
      pad_before.push_back(local_pad_before);
      O_spatial_dims.push_back(local_output_size);
    }
    // The output data dims
    CO = F_CO * G;
    CI = F_CI * G;
    O_dims.push_back(N);
    O_dims.push_back(CO);
    for (size_t i = 0; i < spatial_rank; ++i) {
      O_dims.push_back(O_spatial_dims[i]);
    }
    OC = Contraction(name).outShape(O_dims);
    // The offset data dims
    OFF_C = f_size * spatial_rank * DG;
    OFF_dims.push_back(N);
    OFF_dims.push_back(OFF_C);
    for (size_t i = 0; i < spatial_rank; ++i) {
      OFF_spatial_dims[i] = O_spatial_dims[i];
      OFF_dims.push_back(OFF_spatial_dims[i]);
    }
    OFF.bind_dims(OFF_dims);
    // Input data indexes
    I_idxs.push_back(n);
    I_idxs.push_back((CI / G) * g + ci);
    for (size_t i = 0; i < spatial_rank; ++i) {
      auto index = strides[i] * x[i] + dilations[i] * k[i] - pad_before[i] + OFF(OFF_idxs[i]);
      if (index > I_spatial_dims[i] - 1) {
        index = I_spatial_dims[i] - 1;
      } else if (index < 0) {
        index = 0;
      }
      I_idxs.emplace_back(index);
    }

    std::vector<Constraint> constraints;
    // Offset data indexes
    dg = ((CI / G) * g + ci) / (CI / DG);
    off_c = 0;
    TensorIndex temp("temp");
    for (size_t i = spatial_rank - 1; i >= 0; --i) {
      temp = k[i];
      for (size_t j = i + 1; j < spatial_rank; ++j) {
        temp = temp * F_spatial_dims[j];
      }
      off_c = off_c + temp;
    }
    off_c = off_c + spatial_rank * dg * f_size;
    for (size_t i = 0; i < spatial_rank; ++i) {
      OFF_idxs[i].push_back(n);
      OFF_idxs[i].push_back(off_c + i * f_size);
      for (size_t j = 0; j < spatial_rank; ++j) {
        OFF_idxs[i].emplace_back(x[j]);
      }
    }
    /*
    OFF_idxs.push_back(n);
    OFF_idxs.push_back(off_c);
    for (size_t i = 0; i < spatial_rank; ++i) {
      OFF_idxs.emplace_back(x[i]);
    }
    */
    // Filter indexes
    TensorIndex f_co, f_ci;
    f_co = co;
    f_ci = ci;
    F_idxs.push_back(g);
    F_idxs.push_back(f_co);
    F_idxs.push_back(f_ci);
    for (size_t i = 0; i < spatial_rank; ++i) {
      F_idxs.push_back(k[i]);
    }
    // Output data indexes
    O_idxs.push_back(n);
    O_idxs.push_back((CO / G) * g + co);
    for (size_t i = 0; i < spatial_rank; ++i) {
      O_idxs.push_back(x[i]);
    }
    // Return result
    auto temp_tensor = I;
    for (size_t i = 2; i < spatial_rank + 2; ++i) {
      temp_tensor = temp_tensor.gather(I_idxs[i]).axis(i).interpolationMode(plaidml::edsl::InterpolationMode::LINEAR);
    }
    OC.outAccess(O_idxs).sum(temp_tensor * F(F_idxs)).add_constraints(constraints);
    return Value{OC};
  });
}

}  // namespace PlaidMLPlugin
