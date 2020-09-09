// Copyright 2018 Intel Corporation

#pragma once

#define ELPP_THREAD_SAFE
#define ELPP_CUSTOM_COUT std::cerr
#define ELPP_STL_LOGGING
#define ELPP_LOG_STD_ARRAY
#define ELPP_LOG_UNORDERED_MAP
#define ELPP_LOG_UNORDERED_SET
#define ELPP_NO_LOG_TO_FILE
#define ELPP_DISABLE_DEFAULT_CRASH_HANDLING
#define ELPP_WINSOCK2

#include <easylogging++.h>

#include <string>
#include <vector>

#define IVLOG(N, rest)                                                         \
  do {                                                                         \
    if (VLOG_IS_ON(N)) {                                                       \
      VLOG(N) << rest;                                                         \
    }                                                                          \
  } while (0);

// printf style logging
#define PIVLOG(N, ...)                                                         \
  do {                                                                         \
    if (VLOG_IS_ON(N)) {                                                       \
      el::Loggers::getLogger("default")->verbose(N, __VA_ARGS__);              \
    }                                                                          \
  } while (0);

// VLOGs and writes to a stream.
#define SVLOG(s, N, rest)                                                      \
  do {                                                                         \
    if (VLOG_IS_ON(N)) {                                                       \
      VLOG(N) << rest;                                                         \
    }                                                                          \
    s << rest << "\n";                                                         \
  } while (0);

template <class Iterator>
std::ostream &stringify_collection(std::ostream &os, Iterator it,
                                   Iterator itEnd) {
  os << '[';
  if (it != itEnd) {
    os << *it++;
  }
  for (; it != itEnd; ++it) {
    os << ", " << *it;
  }
  os << ']';
  return os;
}

namespace std {

inline string to_string(const string &x) { return x; }

template <class T>
string to_string(const vector<T> &x) {
  std::stringstream ss;
  stringify_collection(ss, x.begin(), x.end());
  return ss.str();
}

} // namespace std
