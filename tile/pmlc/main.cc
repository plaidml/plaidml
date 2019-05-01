// Copyright 2019, Intel Corporation

#include "base/util/logging.h"
#include "base/util/throw.h"
#include "tile/pmlc/pmlc.h"

namespace fs = boost::filesystem;

int main(int argc, char* argv[]) {
  using vertexai::tile::pmlc::Main;

  try {
    gflags::SetUsageMessage("pmlc <model.tile>");
    START_EASYLOGGINGPP(argc, argv);
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    el::Loggers::reconfigureAllLoggers(vertexai::LogConfigurationFromFlags("default"));
    fs::path input;
    if (argc > 1) {
      input = argv[1];
    }
    Main(input);
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Caught unhandled exception: " << ex.what() << std::endl;
    auto stacktrace = boost::get_error_info<traced>(ex);
    if (stacktrace) {
      std::cerr << *stacktrace << std::endl;
    }
    return -1;
  } catch (...) {
    std::cerr << "Caught unhandled exception" << std::endl;
    return -1;
  }
}
