//
// Copyright 2020 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.

#pragma once

#include <ie_metric_helpers.hpp>

#include <memory>
#include <vector>
#include <string>

#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>

#include "plaidml_state.hpp"

namespace PlaidMLPlugin {

class PlaidMLExecutableNetwork : public InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    using Ptr = std::shared_ptr<PlaidMLExecutableNetwork>;

    PlaidMLExecutableNetwork(InferenceEngine::ICNNNetwork &network, const std::string& configuration_type);
    virtual ~PlaidMLExecutableNetwork() = default;

    InferenceEngine::InferRequestInternal::Ptr
    CreateInferRequestImpl(InferenceEngine::InputsDataMap  networkInputs,
                           InferenceEngine::OutputsDataMap networkOutputs) override;

    void GetMetric(const std::string &name, Parameter &result, ResponseDesc *resp) const override {
        if (name == METRIC_KEY(SUPPORTED_METRICS)) {
            std::vector<std::string> metrics;
            metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
            metrics.push_back(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS));
            result = IE_SET_METRIC(SUPPORTED_METRICS, metrics);
        } else if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
            result = IE_SET_METRIC(OPTIMAL_NUMBER_OF_INFER_REQUESTS, 1);
        } else {
            THROW_IE_EXCEPTION << "Unsupported ExecutableNetwork metric: " << name;
        }
    }

private:
    void InitInputs(InferenceEngine::ICNNNetwork &network);

    // This is a global state that is shared between all infer requests
    std::shared_ptr<State> state_;
};

}  // namespace PlaidMLPlugin
