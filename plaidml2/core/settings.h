// Copyright 2019 Intel Corporation.

#pragma once

#include <map>
#include <string>

#include "plaidml2/core/ffi.h"

namespace plaidml::core {

class Settings {
 public:
  static Settings* Instance();

  const std::map<std::string, std::string>& all() const;
  std::string get(const std::string& key) const;
  void set(const std::string& key, const std::string& value);

  void load();
  void save();

 private:
  Settings();

 private:
  std::map<std::string, std::string> settings_;
};

}  // namespace plaidml::core
