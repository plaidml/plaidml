// Copyright 2018 Intel Corporation.

#include "base/config/config.h"

#include <google/protobuf/descriptor.h>
#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/type_resolver_util.h>

#include <memory>
#include <string>
#include <utility>

#include "base/util/error.h"
#include "base/util/logging.h"
#include "base/util/type_url.h"

namespace gp = google::protobuf;
namespace gpu = google::protobuf::util;

namespace vertexai {

void ParseConfig(const std::string& data, google::protobuf::Message* config) {
  // Try JSON
  std::unique_ptr<gpu::TypeResolver> resolver{
      gpu::NewTypeResolverForDescriptorPool(kTypeVertexAI, gp::DescriptorPool::generated_pool())};
  std::string bin;
  auto status = gpu::JsonToBinaryString(
      resolver.get(), std::string(kTypeVertexAIPrefix) + config->GetDescriptor()->full_name(), data, &bin);
  if (status.ok() && config->ParseFromString(bin)) {
    return;
  }

  // Try serialized proto
  if (config->ParseFromString(data)) {
    return;
  }
  std::string err{"Unable to parse configuration: "};
  status.error_message().AppendToString(&err);
  throw error::InvalidArgument{std::move(err)};
}

std::string SerializeConfig(const gp::Message& config, bool add_whitespace) {
  // N.B. We always serialize configuration as JSON, since that makes it more readable and safer to pass through C
  // NUL-terminated char* parameters. We recognize that this is awfully slow, but if you need high-performance
  // configuration generation and parsing, you're most likely doing something wrong.
  auto resolver = gpu::NewTypeResolverForDescriptorPool(vertexai::kTypeVertexAI, gp::DescriptorPool::generated_pool());

  gpu::JsonOptions options;
  options.add_whitespace = add_whitespace;

  std::string result;
  gpu::BinaryToJsonString(resolver, kTypeVertexAIPrefix + config.GetDescriptor()->full_name(),
                          config.SerializeAsString(), &result, options);

  return result;
}
}  // namespace vertexai
