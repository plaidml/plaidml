// Copyright 2019, Intel Corporation

#pragma once

#include <memory>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "tile/codegen/codegen.pb.h"
#include "tile/lang/compose.h"
#include "tile/stripe/stripe.h"

namespace vertexai {
namespace tile {
namespace pmlc {

class App {
 public:
  boost::program_options::positional_options_description pos_opts;
  boost::program_options::options_description opts;
  boost::program_options::variables_map args;

  static App* Instance();
  bool parse(int argc, char* argv[]);

 private:
  App();
};

vertexai::tile::lang::RunInfo LoadTile(const boost::filesystem::path& filename, bool is_internal);
std::shared_ptr<stripe::Program> Main();
codegen::proto::Configs LoadConfigs();

}  // namespace pmlc
}  // namespace tile
}  // namespace vertexai
