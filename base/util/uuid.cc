// Copyright 2017, Vertex.AI.

#include "base/util/uuid.h"

#include <boost/uuid/name_generator.hpp>
#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/string_generator.hpp>

#include <mutex>
#include <random>

namespace bu = boost::uuids;

namespace vertexai {

const bu::uuid vertexai_uuid_namespace = bu::string_generator()("f39afea4-306c-4fea-8dfa-5b3530618ccb");

bu::uuid GetVertexAIUUID(const char* name) {
  static bu::name_generator vertexai_uuid_gen{vertexai_uuid_namespace};

  return vertexai_uuid_gen(name);
}

bu::uuid GetRandomUUID() {
  static std::mutex random_uuid_mu;
  static boost::mt19937 twister{std::random_device()()};
  static bu::random_generator random_uuid_gen{twister};

  std::lock_guard<std::mutex> lock{random_uuid_mu};
  return random_uuid_gen();
}

}  // namespace vertexai
