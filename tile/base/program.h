// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <list>
#include <map>
#include <memory>
#include <string>

#include "base/context/context.h"
#include "tile/base/buffer.h"

namespace vertexai {
namespace tile {

// Program represents a Tile program that's been compiled by a Platform.
class Program {
 public:
  virtual ~Program() {}

  // Run the program.  The bindings are applied to the proto::Op list that was
  // used to create the Program, in order. Bindings are passed by value so the
  // program can manipulate them as it evaluates a program.
  //
  // Individual buffers may become complete asynchronously to the overall program run; the program itself is complete
  // once the returned future is resolved.
  virtual boost::future<void> Run(const context::Context& ctx, std::map<std::string, std::shared_ptr<Buffer>> inputs,
                                  std::map<std::string, std::shared_ptr<Buffer>> outputs) = 0;
};

}  // namespace tile
}  // namespace vertexai
