// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// NB: IE_SET_METRIC_RETURN requires this header to be included first
#include "ie_metric_helpers.hpp"

#include <memory>

#include "plaidml_plugin.hpp"

// #include "cpp_interfaces/base/ie_plugin_base.hpp" // TODO
// #include "details/caseless.hpp"
// #include "details/ie_cnn_network_tools.h"
#include "ie_plugin_config.hpp"
#include "inference_engine.hpp"

#include "plaidml/exec/exec.h"
#include "plaidml/op/op.h"
#include "plaidml_executable_network.hpp"

using namespace InferenceEngine;

namespace PlaidMLPlugin {

void Engine::GetVersion(const Version*& versionInfo) noexcept {}

ExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(const CNNNetwork& network,
                                                          const std::map<std::string, std::string>& config) {
  auto it = config.find("device");
  const auto& device = it != config.end() ? it->second : "";
  return std::make_shared<PlaidMLExecutableNetwork>(network, device);
}

void Engine::SetConfig(const std::map<std::string, std::string>& config) {
  // Do nothing
}

InferenceEngine::QueryNetworkResult Engine::QueryNetwork(const CNNNetwork& network,
                                                         const std::map<std::string, std::string>& config) const {
  // TODO: do we still need this?
  // std::unordered_set<std::string, details::CaselessHash<std::string>,
  // details::CaselessEq<std::string>>
  //     unsupported_layers = {"detectionoutput", "priorboxclustered",
  //     "regionyolo"};
  // const auto& plugin_name = GetName();
  // auto sorted_layers = CNNNetSortTopologically(network);
  // for (auto layer : sorted_layers) {
  //     auto it = unsupported_layers.find(layer->type);
  //     if (it == unsupported_layers.end()) {
  //         res.supportedLayersMap.insert({layer->name, plugin_name});
  //     }
  // }
  THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
}

Engine::Engine() {
  _pluginName = "PlaidML";
  plaidml::op::init();
  plaidml::exec::init();
}

Parameter Engine::GetMetric(const std::string& name, const std::map<std::string, Parameter>&) const {
  if (name == METRIC_KEY(SUPPORTED_METRICS)) {
    std::vector<std::string> metrics = {
        METRIC_KEY(AVAILABLE_DEVICES),
        METRIC_KEY(SUPPORTED_CONFIG_KEYS),
        METRIC_KEY(SUPPORTED_METRICS),
    };
    IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
  } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
    // FIXME I think it would be more correct to use special CONFIG_KEY
    // is defined in plaidml_config.hpp. But now the bechmark set parameters
    // and we can't use this key there since the plugin is't part of IE.
    // Therefore in the benchmark we pass a string "device" and process it here.
    std::vector<std::string> keys = {
        "device",
    };
    IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, keys);
  } else if (name == METRIC_KEY(AVAILABLE_DEVICES)) {
    std::vector<std::string> devices = {"llvm_cpu"};
    IE_SET_METRIC_RETURN(AVAILABLE_DEVICES, devices);
  }

  throw std::logic_error("Unsupported metric: " + name);
}

}  // namespace PlaidMLPlugin

static const Version version = {{2, 1}, CI_BUILD_NUMBER, "PlaidMLPlugin"};
IE_DEFINE_PLUGIN_CREATE_FUNCTION(PlaidMLPlugin::Engine, version)
