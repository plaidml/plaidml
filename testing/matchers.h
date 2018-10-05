// Copyright 2018 Intel Corporation.

#pragma once

#include <gmock/gmock.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/util/message_differencer.h>
#include <gtest/gtest.h>

#include <array>
#include <iostream>
#include <string>
#include <type_traits>

#include "base/util/compat.h"
#include "plaidml/base/base.h"

namespace testing {

template <typename From, typename To>
class ToProto {
 public:
  typedef vertexai::compat::remove_cv_t<vertexai::compat::remove_reference_t<To>> ToType;
  static ToType Convert(const From& from) { return from; }
};

template <typename To>
class ToProto<const char*, To> {
 public:
  typedef vertexai::compat::remove_cv_t<vertexai::compat::remove_reference_t<To>> ToType;
  static ToType Convert(const char* from) {
    ToType to;
    google::protobuf::TextFormat::ParseFromString(from, &to);
    return to;
  }
};

MATCHER_P(EqualsProto, val, "") {
  auto expected = ToProto<decltype(val), decltype(arg)>::Convert(val);
  std::string output;
  google::protobuf::util::MessageDifferencer diff;
  diff.ReportDifferencesToString(&output);
  if (!diff.Compare(expected, arg)) {
    *result_listener << "\n\nDecoded actual value:\n"
                     << arg.DebugString() << "\nIssues with actual (relative to expected):\n"
                     << output;
    return false;
  }
  return true;
}

MATCHER_P(IsVaiStatus, st, "is vai_status=" + PrintToString(st)) {
  if (arg == st) {
    return true;
  }
  *result_listener << "Last status message: \"" << vai_last_status_str() << "\"";
  return false;
}

}  // namespace testing
