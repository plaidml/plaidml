// Copyright 2018 Intel Corporation.

#include <fstream>
#include <string>

#include "base/util/env.h"
#include "base/util/logging.h"
#include "base/util/runfiles_db.h"
#include "testing/plaidml_config.h"

namespace vertexai {
namespace testing {
namespace {

std::string GetConfigFile() {
  RunfilesDB rdb("com_intel_plaidml", nullptr);
  auto filename = env::Get("PLAIDML_CONFIG_FILE");
  if (!filename.size()) {
    filename = "plaidml/experimental.json";
  }
  return rdb[filename.c_str()];
}

std::string GetConfig() {
  std::string filename = GetConfigFile();
  LOG(INFO) << "Loading: " << filename;
  std::ifstream config_stream(filename);
  config_stream.exceptions(std::ifstream::failbit);
  return std::string(std::istreambuf_iterator<char>(config_stream), std::istreambuf_iterator<char>());
}

}  // namespace

const char* PlaidMLConfig() {
  static std::string config = GetConfig();
  return config.c_str();
}

}  // namespace testing
}  // namespace vertexai
