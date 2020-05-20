// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
/*
#include <utility>
#include <vector>

#include "plaidml_op_fc.hpp"
#include "plaidml_util.hpp"

using namespace plaidml::edsl;
using namespace plaidml::exec;

void OpFc::LoadWeights(State* state) {
    fc_layer_ = dynamic_cast<FullyConnectedLayer*>(layer_.get());

    const auto& in_shape = fc_layer_->insData.front().lock()->getDims();

    Layout l = NC;
    SizeVector w_dims = { fc_layer_->_out_num, in_shape[1] };
    if (in_shape.size() == 4) {
        w_dims.push_back(in_shape[2]);
        w_dims.push_back(in_shape[3]);
        l = OIHW;
    }

    auto weights = fc_layer_->_weights;
    auto biases  = fc_layer_->_biases;

    if (weights) {
        if (l == OIHW || weights->getTensorDesc().getDims().size() == 4) {
            TensorDesc desc(weights->getTensorDesc().getPrecision(), { w_dims[2], w_dims[3], w_dims[1], w_dims[0] },
Layout::ANY); auto binding = util::make_binding(state->device(), desc);

            util::transpose(weights->buffer().as<uint8_t*>(),
                            w_dims,
                            {2, 3, 1, 0},
                            reinterpret_cast<uint8_t*>(binding.buffer.mmap_discard().data()),
                            weights->element_size());

            auto& bindings = state->slot<std::vector<Binding>>()[layer_->name];
            bindings.push_back(std::move(binding));
        } else {
            TensorDesc desc{Precision::FP32, w_dims, l};
            auto binding = util::make_binding(state->device(), desc);
            binding.buffer.copy_from(weights->buffer());

            auto& bindings = state->slot<std::vector<Binding>>()[layer_->name];
            bindings.push_back(std::move(binding));
        }
    }

    if (biases) {
        auto binding = util::make_binding(state->device(), biases->getTensorDesc());
        binding.buffer.copy_from(biases->buffer());

        auto& bindings = state->slot<std::vector<Binding>>()[layer_->name];
        bindings.push_back(std::move(binding));
    }
}

void OpFc::run(const plaidml::edsl::Tensor& I,
               const plaidml::edsl::Tensor& F,
               const plaidml::edsl::Tensor& B,
               plaidml::edsl::Tensor& O) {
    const auto& in_shape = fc_layer_->insData.front().lock()->getDims();

    IE_ASSERT((in_shape.size() == 2 || in_shape.size() == 4) &&
              "Currently only 2D and 4D fullyconnected is supported");

    TensorDim N, CI, CO, HI, WI;
    TensorIndex n, o, h, c, w;

    if (in_shape.size() == 2) {
        I.bind_dims(N, CI);
        F.bind_dims(CO, CI);
        B.bind_dims(CO);

        O = TensorOutput(N, CO);
        O(n, o) += I(n, h) * F(o, h);
        O(n, o) = O(n, o) + B(o);
    } else if (in_shape.size() == 4) {
        I.bind_dims(N, HI, WI, CI);
        F.bind_dims(HI, WI, CI, CO);
        B.bind_dims(CO);

        O = TensorOutput(N, CO);
        O(n, o) += I(n, h, w, c) * F(h, w, c, o);
        O(n, o) = O(n, o) + B(o);
    }
}
*/