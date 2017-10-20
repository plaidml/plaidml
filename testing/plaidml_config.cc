// Copyright Vertex.AI.

#include <fstream>
#include <string>

#include "base/util/logging.h"
#include "testing/plaidml_config.h"
#include "testing/runfiles_db.h"

namespace vertexai {
namespace testing {
namespace {

std::string GetConfigFile() {
  RunfilesDB rdb("vertexai_plaidml", nullptr);
  const char* filename = std::getenv("PLAIDML_CONFIG_FILE");
  if (!filename) {
    filename = "testing/tile_generic_cpu.json";
  }
  return rdb[filename];
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
