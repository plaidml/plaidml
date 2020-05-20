//
// Copyright (C) 2020 Intel Corporation.
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
//

// NB: MAGIC BUT THIS HEADER SHOULD BE INCLUDED FIRST
#include "ie_metric_helpers.hpp"

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "plaidml_executable_network.hpp"
#include "plaidml_plugin.hpp"

#include "details/caseless.hpp"
#include "inference_engine.hpp"

#include <cpp_interfaces/base/ie_plugin_base.hpp>
#include <details/ie_cnn_network_tools.h>

#include <ie_plugin_config.hpp>

using namespace InferenceEngine;
using namespace InferenceEngine::PluginConfigParams;
using namespace PlaidMLPlugin;
using namespace std;

void Engine::GetVersion(const Version*& versionInfo) noexcept {}

InferenceEngine::ExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(
    const InferenceEngine::ICore* /* core */, InferenceEngine::ICNNNetwork& network,
    const std::map<std::string, std::string>& config) {
  auto it = config.find("device");
  const auto& configuration_type = it != config.end() ? it->second : "";
  return std::make_shared<PlaidMLExecutableNetwork>(network, configuration_type);
}

void Engine::SetConfig(const std::map<std::string, std::string>& config) { /* Do nothing */ }

void Engine::AddExtension(InferenceEngine::IExtensionPtr extension) {
  throw std::logic_error("AddExtension not implemented!!!");
}

void Engine::QueryNetwork(const InferenceEngine::ICNNNetwork& network, const std::map<std::string, std::string>& config,
                          InferenceEngine::QueryNetworkResult& res) const {
  std::unordered_set<std::string, details::CaselessHash<std::string>, details::CaselessEq<std::string>>
      unsupported_layers = {"detectionoutput", "priorboxclustered", "regionyolo"};
  const auto& plugin_name = GetName();
  auto sorted_layers = CNNNetSortTopologically(network);
  for (auto layer : sorted_layers) {
    auto it = unsupported_layers.find(layer->type);
    if (it == unsupported_layers.end()) {
      res.supportedLayersMap.insert({layer->name, plugin_name});
    }
  }
}

void Engine::SetLogCallback(InferenceEngine::IErrorListener& listener) {
  throw std::logic_error("SetLogCallback not implemented!!!");
}

IE_SUPPRESS_DEPRECATED_START

INFERENCE_PLUGIN_API(StatusCode) CreatePluginEngine(IInferencePlugin*& plugin, ResponseDesc* resp) noexcept {
  try {
    plugin = make_ie_compatible_plugin({{1, 6}, CI_BUILD_NUMBER, "PlaidMLPlugin"}, std::make_shared<Engine>());
    return OK;
  } catch (std::exception& ex) {
    return DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();
  }
}

InferenceEngine::Parameter PlaidMLPlugin::Engine::GetMetric(
    const std::string& name, const std::map<std::string, InferenceEngine::Parameter>&) const {
  if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
    std::vector<std::string> metrics;
    // FIXME I think it would be more correct to use special CONFIG_KEY
    // is defined in plaidml_config.hpp. But now the bechmark set parameters
    // and we can't use this key there since the plugin is't part of IE.
    // Therefore in the benchmark we pass a string "device" and process it here.
    metrics.push_back("device");
    IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, metrics);
  }
  throw std::logic_error("Unsupported metric name");
}

IE_SUPPRESS_DEPRECATED_END
