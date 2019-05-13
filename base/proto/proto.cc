// Copyright 2019, Intel Corp.

#include "base/proto/proto.h"

#include "base/util/type_url.h"

namespace gp = google::protobuf;

namespace vertexai {

const gp::FieldDescriptor* ProtoFinder::FindExtension(gp::Message* message, const std::string& name) const {
  return nullptr;
}

const gp::Descriptor* ProtoFinder::FindAnyType(const gp::Message& /* message */, const std::string& prefix,
                                               const std::string& name) const {
  if (prefix != kTypeVertexAIPrefix) {
    return nullptr;
  }
  return gp::DescriptorPool::generated_pool()->FindMessageTypeByName(name);
}

}  // namespace vertexai
