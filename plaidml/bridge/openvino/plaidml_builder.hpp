// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_icnn_network.hpp"
#include "plaidml/core/core.h"

namespace PlaidMLPlugin {

plaidml::Program buildProgram(const InferenceEngine::ICNNNetwork& network);

}  // namespace PlaidMLPlugin
