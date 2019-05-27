// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <map>
#include <memory>
#include <string>

#include "tile/base/program.h"
#include "tile/proto/tile.pb.h"

namespace vertexai {
namespace tile {
namespace targets {
namespace cpu {
class Native;
}  // namespace cpu
}  // namespace targets
}  // namespace tile
}  // namespace vertexai

namespace vertexai {
namespace tile {
namespace stripejit {

class Program final : public tile::Program {
 public:
  Program(const context::Context& ctx, const tile::proto::Program& program, ConstBufferManager* const_bufs);
  ~Program();

  boost::future<void> Run(const context::Context& ctx, std::map<std::string, std::shared_ptr<tile::Buffer>> inputs,
                          std::map<std::string, std::shared_ptr<tile::Buffer>> outputs) final;

 private:
  std::unique_ptr<tile::targets::cpu::Native> executable_;
};

}  // namespace stripejit
}  // namespace tile
}  // namespace vertexai
