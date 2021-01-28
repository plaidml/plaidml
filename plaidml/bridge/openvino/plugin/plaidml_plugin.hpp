// Copyright (C) 2021 Intel Corporation
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
  Engine();
  virtual ~Engine() = default;

  void GetVersion(const InferenceEngine::Version*& versionInfo) noexcept;

  InferenceEngine::ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(
      const InferenceEngine::CNNNetwork& network, const std::map<std::string, std::string>& config) override;

  InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                   const std::map<std::string, std::string>& config) const override;

  void SetConfig(const std::map<std::string, std::string>& config) override;

  InferenceEngine::Parameter GetMetric(const std::string& name,
                                       const std::map<std::string, InferenceEngine::Parameter>&) const override;
};

}  // namespace PlaidMLPlugin
