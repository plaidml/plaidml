// Copyright 2018, Intel Corporation

#pragma once

#include <boost/filesystem.hpp>

#include "tile/codegen/codegen.pb.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

struct OptimizeOptions {
  bool dump_passes = false;
  bool dump_code = false;
  boost::filesystem::path dbg_dir;
};

using Passes = google::protobuf::RepeatedPtrField<proto::Pass>;

void Optimize(stripe::Block* block, const Passes& passes, const OptimizeOptions& options);

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
