// Copyright 2018 Intel Corporation.

#pragma once

#include <google/protobuf/message.h>

#include <string>

namespace vertexai {

void ParseConfig(const std::string& data, google::protobuf::Message* config);

template <typename ProtoConfig>
ProtoConfig ParseConfig(const std::string& data) {
  ProtoConfig config;
  ParseConfig(data, &config);
  return config;
}

std::string SerializeConfig(const google::protobuf::Message& config, bool add_whitespace = false);

}  // namespace vertexai
