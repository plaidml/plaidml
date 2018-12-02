// Copyright 2017-2018 Intel Corporation.

// This is an overly-simplistic utility for doing some simple printf
// timing debugging.  It's not useful for extremely short time periods
// or lots of calls, and it'd be vastly better to use a real profiler,
// but if you're trying to understand where milliseconds are going
// within a single function and you have a good repo, it can be
// useful.
//
// To use it:
//   * pdebug::Trace ts{"WhatYouAreDoing"};
//   * ts.Capture("some_useful_string")  // At interesting timepoints
//
// When the trace is destroyed, it will print its capture via
// LOG(INFO).  This makes it easy to tell when it's present, so you
// can remove it before checkin; it's not designed for production
// usage.

#pragma once

#include <chrono>
#include <list>
#include <string>
#include <utility>

#include "base/util/logging.h"

namespace vertexai {
namespace pdebug {

class Trace {
 public:
  explicit Trace(std::string what) : what_{std::move(what)} {}

  ~Trace() {
    LOG(INFO) << what_ << ":";
    auto it = ts_.begin();
    auto prev = *it++;
    while (it != ts_.end()) {
      auto next = *it++;
      LOG(INFO) << "  " << Timepoint::Delta(prev, next);
      prev = next;
    }
  }

  void Capture(std::string str) { ts_.emplace_back(std::move(str)); }

 private:
  class Timepoint {
   public:
    explicit Timepoint(std::string name) : ts_{std::chrono::high_resolution_clock::now()}, name_{std::move(name)} {}

    static std::string Delta(const Timepoint& prev, const Timepoint& next) {
      return std::to_string(std::chrono::duration<double>(next.ts_ - prev.ts_).count()) + " : " + prev.name_ + " - " +
             next.name_;
    }

   private:
    std::chrono::time_point<std::chrono::high_resolution_clock> ts_;
    std::string name_;
  };

  std::string what_;
  std::list<Timepoint> ts_;
};

}  // namespace pdebug
}  // namespace vertexai
