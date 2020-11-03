// Copyright 2019 Intel Corporation.

#include "plaidml/core/settings.h"

#include <fstream>

#include "llvm/Support/FileSystem.h"

#include "plaidml/core/internal.h"
#include "pmlc/util/env.h"
#include "pmlc/util/logging.h"

using plaidml::core::ffi_wrap;
using plaidml::core::ffi_wrap_void;
namespace fs = llvm::sys::fs;

namespace plaidml::core {

Settings::Settings() {
  std::string path = pmlc::util::getEnvVar("PLAIDML_SETTINGS");
  if (path.empty()) {
    llvm::SmallString<64> expandedPath;
    fs::expand_tilde("~/.plaidml2", expandedPath);
    path = static_cast<std::string>(expandedPath);
  }
  settings_.emplace("PLAIDML_SETTINGS", path);
}

Settings* Settings::Instance() {
  static Settings settings;
  return &settings;
}

const std::map<std::string, std::string>& Settings::all() const {  //
  return settings_;
}

std::string Settings::get(const std::string& key) const {
  auto env_var = pmlc::util::getEnvVar(key);
  if (env_var.size()) {
    return env_var;
  }
  auto it = settings_.find(key);
  if (it == settings_.end()) {
    return "";
  }
  return it->second;
}

void Settings::set(const std::string& key, const std::string& value) {
  settings_[key] = value;
  pmlc::util::setEnvVar(key, value);
}

void Settings::load() {
  auto settings_path = get("PLAIDML_SETTINGS");
  if (!fs::exists(settings_path)) {
    LOG(WARNING) << "No PlaidML settings found.";
    return;
  }
  settings_.clear();
  std::ifstream file(settings_path);
  for (std::string line; std::getline(file, line);) {
    auto pos = line.find('=');
    if (pos != std::string::npos) {
      auto key = line.substr(0, pos);
      auto value = line.substr(pos + 1);
      IVLOG(1, key << " = " << value);
      auto env_var = pmlc::util::getEnvVar(key);
      if (env_var.size()) {
        value = env_var;
      }
      set(key, value);
    }
  }
  settings_["PLAIDML_SETTINGS"] = settings_path;
}

void Settings::save() {
  auto settings_path = get("PLAIDML_SETTINGS");
  std::ofstream file(settings_path);
  for (const auto& kvp : settings_) {
    if (kvp.first != "PLAIDML_SETTINGS") {
      file << kvp.first << "=" << kvp.second << std::endl;
    }
  }
}

}  // namespace plaidml::core
