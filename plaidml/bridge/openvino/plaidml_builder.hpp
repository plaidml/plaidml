// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "ie_icnn_network.hpp"
#include "ngraph/function.hpp"
#include "plaidml/core/core.h"

namespace PlaidMLPlugin {

plaidml::Program buildProgram(const std::shared_ptr<const ngraph::Function>& func, const std::string& netName,
                              const InferenceEngine::InputsDataMap& inputsInfo,
                              const InferenceEngine::OutputsDataMap& outputsInfo);

plaidml::Program buildNodeProgram(const std::shared_ptr<ngraph::Node>& node);

}  // namespace PlaidMLPlugin
