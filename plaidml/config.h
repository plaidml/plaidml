// Copyright 2019 Intel Corporation.

#pragma once

#include <string>

namespace vertexai {
namespace plaidml {
namespace config {

struct Config {
  std::string data;
  std::string source;
};

// Gets the configuration to use for the PlaidML library.
Config Get();

}  // namespace config
}  // namespace plaidml
}  // namespace vertexai
