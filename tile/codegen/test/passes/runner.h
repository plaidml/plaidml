// Copyright 2019, Intel Corporation
#pragma once

#include <boost/filesystem.hpp>

namespace vertexai {
namespace tile {
namespace codegen {
namespace test {
namespace passes {

bool VerifyPasses(const boost::filesystem::path& passes_dir);

}  // namespace passes
}  // namespace test
}  // namespace codegen
}  // namespace tile
}  // namespace vertexai
