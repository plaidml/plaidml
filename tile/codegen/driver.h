// Copyright 2018, Intel Corporation

#pragma once

#include <boost/filesystem.hpp>

#include "tile/codegen/codegen.pb.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

struct OptimizeOptions {
  bool dump_passes;
  boost::filesystem::path dbg_dir;
};

void Optimize(stripe::Block* block, const proto::Config& cfg, const OptimizeOptions& options);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
