// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "plaidml_state.hpp"
#include "plaidml_util.hpp"

using namespace PlaidMLPlugin;

State::State(const std::string& configuration_type) {
  dev_conf.dev_ = util::find_device(configuration_type);
  dev_conf.tar_ = util::find_target(configuration_type);
}
