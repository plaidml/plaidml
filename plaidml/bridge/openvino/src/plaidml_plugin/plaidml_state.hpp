// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ie_blob.h>
#include <ie_layouts.h>

#include "plaidml/core/core.h"
#include "plaidml/edsl/edsl.h"
#include "plaidml/exec/exec.h"

#include "plaidml_util.hpp"

using namespace InferenceEngine;

namespace PlaidMLPlugin {

struct DevConfig {
  std::string dev_;
  std::string tar_;
};

class State {
  using Mag = util::Magazine<plaidml::edsl::Tensor, std::vector<plaidml::exec::Binding>>;

 public:
  explicit State(const std::string& configuration_type);

  template <typename T>
  Mag::MapT<T>& slot() {
    return mag_.slot<T>();
  }

  template <typename T>
  const Mag::MapT<T>& slot() const {
    return mag_.slot<T>();
  }

  const std::string& device() const { return dev_conf.dev_; }
  const std::string& target() const { return dev_conf.tar_; }

 private:
  Mag mag_;
  DevConfig dev_conf;
};

}  // namespace PlaidMLPlugin
