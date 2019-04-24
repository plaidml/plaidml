// Copyright 2019, Intel Corporation

#pragma once

#include <gflags/gflags.h>

#include <memory>

#include <boost/filesystem.hpp>

#include "tile/stripe/stripe.h"

DECLARE_string(config);
DECLARE_bool(dump_passes);
DECLARE_string(outdir);

namespace vertexai {
namespace tile {
namespace pmlc {

std::shared_ptr<stripe::Program> Main(const boost::filesystem::path& filename);

}  // namespace pmlc
}  // namespace tile
}  // namespace vertexai
