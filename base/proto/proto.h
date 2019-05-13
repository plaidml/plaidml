// Copyright 2019, Intel Corp.

#pragma once

#include <google/protobuf/text_format.h>

#include <string>

namespace vertexai {

class ProtoFinder final : public google::protobuf::TextFormat::Finder {
 public:
  const google::protobuf::FieldDescriptor* FindExtension(google::protobuf::Message* message,
                                                         const std::string& name) const final;
  const google::protobuf::Descriptor* FindAnyType(const google::protobuf::Message& message, const std::string& prefix,
                                                  const std::string& name) const final;
};

template <typename P>
P ParseProtoText(const std::string& txt) {
  P proto;
  google::protobuf::TextFormat::Parser parser;
  ProtoFinder finder;
  parser.SetFinder(&finder);
  parser.ParseFromString(txt, &proto);
  return proto;
}

}  // namespace vertexai
