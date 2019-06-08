// Copyright 2018, Intel Corporation

#pragma once

#include <string>

#include <boost/filesystem.hpp>

#include "tile/codegen/codegen.pb.h"
#include "tile/codegen/compile_pass.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace codegen {

struct OptimizeOptions {
  bool dump_passes = false;
  bool dump_passes_proto = false;
  bool dump_code = false;
  boost::filesystem::path dbg_dir;
};

using Passes = google::protobuf::RepeatedPtrField<proto::Pass>;

void Optimize(CompilerState* state, const Passes& passes, const OptimizeOptions& options);

struct Configs {
  static void Register(const std::string& name, const std::string& pb_bytes);
  static proto::Config Resolve(const std::string& name);
};

}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
