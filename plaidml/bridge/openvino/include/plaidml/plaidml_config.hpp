// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header for advanced hardware related properties for PlaidML plugin
 *        To use in SetConfig() method of plugins
 *
 * @file ie_plugin_config.hpp
 */
#pragma once

#include "ie_plugin_config.hpp"

namespace InferenceEngine {

namespace PlaidMLConfigParams {

/**
 * @brief shortcut for defining configuration keys
 */
#define PlaidML_CONFIG_KEY(name) InferenceEngine::PlaidMLConfigParams::_CONFIG_KEY(PlaidML_##name)
#define DECLARE_PlaidML_CONFIG_KEY(name) DECLARE_CONFIG_KEY(PlaidML_##name)
#define DECLARE_PlaidML_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(PlaidML_##name)

}  // namespace PlaidMLConfigParams
}  // namespace InferenceEngine
