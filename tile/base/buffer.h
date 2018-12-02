// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include <boost/thread/future.hpp>

#include "base/context/context.h"

namespace vertexai {
namespace tile {

// View is an abstract base class representing a mapped view of a buffer's memory.
class View {
 public:
  virtual ~View() {}

  // Writes the contents of the view back to the device (if necessary).  After this call is made, the caller may
  // immediately issue subsequent operations that will observe the view's current contents, and may safely delete the
  // view.  The caller must not access the view's data after this call is made; the implementation is allowed to unmap
  // it.
  virtual void WriteBack(const context::Context& ctx) = 0;

  char* data() { return data_; }
  const char* data() const { return data_; }
  std::size_t size() const { return size_; }

  // Helper accessors
  char& operator[](std::size_t pos) { return data_[pos]; }
  const char& operator[](std::size_t pos) const { return data_[pos]; }
  char* begin() { return data_; }
  const char* begin() const { return data_; }
  const char* cbegin() const { return data_; }
  char* end() { return data_ + size_; }
  const char* end() const { return data_ + size_; }
  const char* cend() const { return data_ + size_; }

  std::string str() const { return std::string(cbegin(), cend()); }

 protected:
  View() {}
  View(char* data, std::size_t size) : data_{data}, size_{size} {}

  void set_contents(char* data, std::size_t size) {
    data_ = data;
    size_ = size;
  }

 private:
  char* data_ = nullptr;
  std::size_t size_ = 0;
};

// Buffer represents a buffer residing on some Platform.
class Buffer {
 public:
  virtual ~Buffer() {}

  virtual std::uint64_t size() const = 0;

  // Asynchronously maps a read/write view of a buffer.  All views of a buffer must be deleted before the buffer is
  // passed to Program::Run.  Note that this API may raise an error synchronously (e.g. under low memory conditions) or
  // asynchronously (e.g. a problem with the underlying device, or with the calls that created the buffer's contents).
  virtual boost::future<std::unique_ptr<View>> MapCurrent(const context::Context& ctx) = 0;

  // Synchronously maps a read/write view of a buffer, optionally (implementation-specific) discarding the buffer's
  // existing contents.
  virtual std::unique_ptr<View> MapDiscard(const context::Context& ctx) = 0;
};

}  // namespace tile
}  // namespace vertexai
