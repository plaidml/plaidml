// Copyright 2019 Intel Corporation.

#include "plaidml/config.h"

#include <boost/dll/runtime_symbol_info.hpp>
#include <boost/filesystem.hpp>

#include "base/config/config.h"
#include "base/util/env.h"
#include "base/util/runfiles_db.h"

namespace vertexai {
namespace plaidml {
namespace config {

namespace {

const char* kPlaidMLExperimental = "PLAIDML_EXPERIMENTAL";
const char* kPlaidMLDefaultConfigBasename = "plaidml/config.json";
const char* kPlaidMLExperimentalConfigBasename = "plaidml/experimental.json";

}  // namespace

Config Get() {
  static vertexai::RunfilesDB runfiles_db{"com_intel_plaidml"};
  Config config;
  std::string exp = vertexai::env::Get(kPlaidMLExperimental);
  if (!exp.empty() && exp != "0") {
    config.source = kPlaidMLExperimentalConfigBasename;
  } else {
    config.source = kPlaidMLDefaultConfigBasename;
  }
  std::string translated = runfiles_db[config.source.c_str()];
  std::ifstream cfs{translated};

  config.data.assign(std::istreambuf_iterator<char>(cfs), std::istreambuf_iterator<char>());

  return config;
}

}  // namespace config
}  // namespace plaidml
}  // namespace vertexai
