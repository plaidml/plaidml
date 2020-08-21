// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// NB: IE_SET_METRIC_RETURN requires this header to be included first
#include "ie_metric_helpers.hpp"

#include "plaidml_plugin.hpp"

#include <memory>

#include "cpp_interfaces/base/ie_plugin_base.hpp"
// #include "details/caseless.hpp"
// #include "details/ie_cnn_network_tools.h"
#include "ie_plugin_config.hpp"
#include "inference_engine.hpp"

#include "plaidml/exec/exec.h"
#include "plaidml/op/op.h"
#include "plaidml_executable_network.hpp"
#include "pmlc/util/logging.h"

using namespace InferenceEngine;  // NOLINT[build/namespaces]

namespace PlaidMLPlugin {

void Engine::GetVersion(const Version*& versionInfo) noexcept {}

ExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(const ICNNNetwork& network,
                                                          const std::map<std::string, std::string>& config) {
  IVLOG(1, "Engine::LoadExeNetworkImpl> config: " << config);
  auto it = config.find("device");
  const auto& device = it != config.end() ? it->second : "";
  return std::make_shared<PlaidMLExecutableNetwork>(network, device);
}

void Engine::SetConfig(const std::map<std::string, std::string>& config) {
  IVLOG(1, "Engine::SetConfig>");
  // Do nothing
}

void Engine::QueryNetwork(const ICNNNetwork& network, const std::map<std::string, std::string>& config,
                          QueryNetworkResult& result) const {
  IVLOG(1, "Engine::QueryNetwork>");
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
}

IE_SUPPRESS_DEPRECATED_START

INFERENCE_PLUGIN_API(StatusCode)
CreatePluginEngine(IInferencePlugin*& plugin, ResponseDesc* resp) noexcept {
  try {
    plaidml::op::init();
    plaidml::exec::init();

    IVLOG(1, "CreatePluginEngine>");
    plugin = make_ie_compatible_plugin({{1, 6}, CI_BUILD_NUMBER, "PlaidMLPlugin"}, std::make_shared<Engine>());
    return OK;
  } catch (std::exception& ex) {
    return DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();
  }
}

Parameter Engine::GetMetric(const std::string& name, const std::map<std::string, Parameter>&) const {
  IVLOG(1, "Engine::GetMetric> " << name);
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

IE_SUPPRESS_DEPRECATED_END

}  // namespace PlaidMLPlugin
