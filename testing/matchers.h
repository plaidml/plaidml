// Copyright 2018 Intel Corporation.

#pragma once

#include <gmock/gmock.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/util/json_util.h>
#include <google/protobuf/util/message_differencer.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

#include "base/util/compat.h"
#include "plaidml/base/base.h"

namespace testing {

MATCHER_P(EqualsProtoJson, json, "") {
  typedef vertexai::compat::remove_cv_t<vertexai::compat::remove_reference_t<decltype(arg)>> ProtoType;
  ProtoType expected;
  google::protobuf::util::JsonStringToMessage(json, &expected);
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

MATCHER_P(EqualsProtoText, text, "") {
  typedef vertexai::compat::remove_cv_t<vertexai::compat::remove_reference_t<decltype(arg)>> ProtoType;
  ProtoType expected;
  google::protobuf::TextFormat::ParseFromString(text, &expected);
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

inline std::vector<std::string> GetLines(const std::string& str) {
  std::vector<std::string> ret;
  std::istringstream input(str);
  for (std::string line; std::getline(input, line);) {
    ret.push_back(line);
  }
  return ret;
}

MATCHER_P(LinesEq, str, "") {
  if (str == arg) {
    return true;
  }

  std::vector<std::string> lines_a = GetLines(str);
  std::vector<std::string> lines_b = GetLines(arg);

  for (size_t i = 0; i < std::min(lines_a.size(), lines_b.size()); i++) {
    if (lines_a[i] != lines_b[i]) {
      *result_listener << "\nMismatch on line " << i << ":\n"
                       << "expected: " << lines_a[i] << "\n"
                       << "actual  : " << lines_b[i];
    }
  }

  return false;
}

}  // namespace testing
