// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "tile/base/buffer.h"

namespace vertexai {
namespace tile {
namespace stripejit {

class Buffer : public tile::Buffer, public std::enable_shared_from_this<Buffer> {
 public:
  explicit Buffer(std::uint64_t size);

  std::uint64_t size() const final;

  boost::future<std::unique_ptr<tile::View>> MapCurrent(const context::Context& ctx) final;

  std::unique_ptr<tile::View> MapDiscard(const context::Context& ctx) final;

 private:
  std::vector<char> data_;
};

}  // namespace stripejit
}  // namespace tile
}  // namespace vertexai
