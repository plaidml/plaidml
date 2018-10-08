// Copyright 2018 Intel Corporation.
//
// This is the Vertex.AI common C++ interface, which provides a higher level object
// oriented wrapper on top of the Vertex.AI common C API.

#pragma once

#include <exception>
#include <memory>
#include <string>
#include <utility>

#include "plaidml/base/base.h"

namespace vertexai {

class vai_exception : public std::runtime_error {
 public:
  vai_exception(vai_status status, const std::string& what) : std::runtime_error(what), status_(status) {}
  vai_status status() { return status_; }

  template <typename T>
  static void check_and_throw(const T& good) {
    if (good) {
      return;
    }
    vai_status status = vai_last_status();
    std::string err = vai_last_status_str();
    vai_clear_status();
    throw vai_exception{status, err.c_str()};
  }

  static std::exception_ptr current() noexcept {
    try {
      vai_status status = vai_last_status();
      std::string err = vai_last_status_str();
      vai_clear_status();
      return std::make_exception_ptr(vai_exception{status, err.c_str()});
    } catch (...) {
      return std::current_exception();
    }
  }

 private:
  vai_status status_;
};

class ctx final {
 public:
  ctx() : ctx_{vai_alloc_ctx()} {
    if (!ctx_) {
      throw std::bad_alloc();
    }
  }

  explicit ctx(std::unique_ptr<vai_ctx> ctx) : ctx_{std::move(ctx)} {
    if (!ctx_) {
      throw std::bad_alloc();
    }
  }

  vai_ctx* get_ctx() const { return ctx_.get(); }

 private:
  std::unique_ptr<vai_ctx> ctx_;
};

}  // namespace vertexai
