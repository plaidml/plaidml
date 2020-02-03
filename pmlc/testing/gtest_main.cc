#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <cstdlib>

#include "pmlc/util/logging.h"

int main(int argc, char *argv[]) {
  gflags::SetUsageMessage(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  START_EASYLOGGINGPP(argc, argv);

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc != 1) {
    gflags::ShowUsageWithFlags(argv[0]);
    return EXIT_FAILURE;
  }

  el::Loggers::reconfigureAllLoggers(
      pmlc::util::LogConfigurationFromFlags("default"));
  try {
    return RUN_ALL_TESTS();
  } catch (const std::exception &ex) {
    std::cerr << "Caught unhandled exception: " << ex.what() << std::endl;
    return EXIT_FAILURE;
  } catch (...) {
    std::cerr << "Caught unhandled exception" << std::endl;
    return EXIT_FAILURE;
  }
}
