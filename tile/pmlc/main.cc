// Copyright 2019, Intel Corporation

#include "base/util/logging.h"
#include "base/util/throw.h"
#include "tile/pmlc/pmlc.h"

int main(int argc, char* argv[]) {
  using vertexai::tile::pmlc::Main;

  try {
    START_EASYLOGGINGPP(argc, argv);
    Main();
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
