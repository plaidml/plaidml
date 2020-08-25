// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>

#include "cpp_interfaces/impl/ie_plugin_internal.hpp"

namespace PlaidMLPlugin {

class Engine : public InferenceEngine::InferencePluginInternal {
 public:
  Engine() = default;
  virtual ~Engine() = default;

  void GetVersion(const InferenceEngine::Version*& versionInfo) noexcept;

  InferenceEngine::ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(
      const InferenceEngine::ICNNNetwork& network, const std::map<std::string, std::string>& config) override;

  void QueryNetwork(const InferenceEngine::ICNNNetwork& network, const std::map<std::string, std::string>& config,
                    InferenceEngine::QueryNetworkResult& res) const override;

  void SetConfig(const std::map<std::string, std::string>& config) override;

  InferenceEngine::Parameter GetMetric(const std::string& name,
                                       const std::map<std::string, InferenceEngine::Parameter>&) const override;
};

}  // namespace PlaidMLPlugin
