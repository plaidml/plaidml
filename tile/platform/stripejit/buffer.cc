// Copyright 2017-2018 Intel Corporation.

#include "tile/platform/stripejit/buffer.h"

#include <utility>

#include <boost/thread/future.hpp>

namespace vertexai {
namespace tile {
namespace stripejit {

class View final : public tile::View {
  typedef tile::View inherited;

 public:
  View(char* data, std::size_t size) : inherited::View(data, size) {}

  void WriteBack(const context::Context& ctx) final {}
};

Buffer::Buffer(std::uint64_t size) : data_(size, '\0') {}

std::uint64_t Buffer::size() const { return data_.size(); }

boost::future<std::unique_ptr<tile::View>> Buffer::MapCurrent(const context::Context& ctx) {
  std::unique_ptr<tile::View> view(new View(data_.data(), data_.size()));
  return boost::make_ready_future(std::move(view));
}

std::unique_ptr<tile::View> Buffer::MapDiscard(const context::Context& ctx) {
  std::unique_ptr<tile::View> view(new View(data_.data(), data_.size()));
  return view;
}

}  // namespace stripejit
}  // namespace tile
}  // namespace vertexai
