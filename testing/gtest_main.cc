#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <cstdlib>

#include "base/util/logging.h"

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  START_EASYLOGGINGPP(argc, argv);

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc != 1) {
    gflags::ShowUsageWithFlags(argv[0]);
    return EXIT_FAILURE;
  }

  el::Loggers::reconfigureAllLoggers(vertexai::LogConfigurationFromFlags("default"));
  return RUN_ALL_TESTS();
}
