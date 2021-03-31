// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>
#include <vector>

#include "pmlc/util/shape.h"

namespace pmlc::util {

class Buffer;
using BufferPtr = std::shared_ptr<Buffer>;

class Buffer {
public:
  virtual ~Buffer() {}
  virtual size_t size() const = 0;
  virtual char *data() = 0;
  virtual BufferPtr clone() = 0;
  virtual TensorShape shape() = 0;
};

// A simple buffer backed by a std::vector
class SimpleBuffer : public Buffer,
                     public std::enable_shared_from_this<SimpleBuffer> {
public:
  explicit SimpleBuffer(const TensorShape &shape)
      : shape_(shape), data_(shape.getByteSize()) {}

  SimpleBuffer(const TensorShape &shape, const std::vector<char> &data)
      : shape_(shape), data_(data) {}

  size_t size() const final { return data_.size(); }

  char *data() final { return data_.data(); }

  BufferPtr clone() final {
    return std::make_shared<SimpleBuffer>(shape_, data_);
  }

  TensorShape shape() final { return shape_; }

private:
  TensorShape shape_;
  std::vector<char> data_;
};

// An adopted buffer owned by the user.
class AdoptedBuffer : public Buffer,
                      public std::enable_shared_from_this<AdoptedBuffer> {
public:
  AdoptedBuffer(const TensorShape &shape, size_t size, char *data)
      : shape_(shape), size_(size), data_(data) {}

  size_t size() const final { return size_; }

  char *data() final { return data_; }

  BufferPtr clone() final {
    return std::make_shared<AdoptedBuffer>(shape_, size_, data_);
  }

  TensorShape shape() final { return shape_; }

private:
  TensorShape shape_;
  size_t size_;
  char *data_;
};

class RawBuffer : public Buffer,
                  public std::enable_shared_from_this<RawBuffer> {
public:
  size_t size() const final { return vec.size(); }

  char *data() final { return vec.data(); }

  BufferPtr clone() final {
    auto ret = std::make_shared<RawBuffer>();
    ret->vec = vec;
    return ret;
  }

  TensorShape shape() final {
    return TensorShape(DataType::ui8, {static_cast<int64_t>(size())});
  }

  std::vector<char> vec;
};

} // namespace pmlc::util
