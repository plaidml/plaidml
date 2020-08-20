#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <sstream>

#include "pmlc/util/logging.h"

DEFINE_string(skip_test_file, "", "A file containing a list of test patterns to skip");

int main(int argc, char *argv[]) {
  gflags::SetUsageMessage(argv[0]);
  testing::InitGoogleTest(&argc, argv);
  START_EASYLOGGINGPP(argc, argv);

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc != 1) {
    gflags::ShowUsageWithFlags(argv[0]);
    return EXIT_FAILURE;
  }

  if (FLAGS_skip_test_file.size()) {
    // If there's a skip test file, we're going to take each pattern it contains,
    // and add those patterns to gtest's filter's negative pattern list.

    // To start, see whether there's a '-' in the existing filter.  If
    // there isn't, we'll need to add one (to indicate that these are
    // negative tests); if there is, we should use ':'.  Regardless,
    // each subsequent added pattern is separated by ':'.
    char sep = ':';
    if (testing::FLAGS_gtest_filter.find('-') == std::string::npos) {
      sep = '-';
    }

    // Process the skip pattern file, building the flags pattern.
    std::ifstream fs{FLAGS_skip_test_file};
    if (!fs) {
      std::cerr << "Unable to open skip test file '" << FLAGS_skip_test_file << "'\n";
      return EXIT_FAILURE;
    }

    std::ostringstream ss{testing::FLAGS_gtest_filter};

    for (std::string line; std::getline(fs >> std::ws, line);) {
      // Erase trailing whitespace
      line.erase(std::find_if_not(line.rbegin(), line.rend(), [](int c) -> bool { return std::isspace(c); }).base(), line.end());
      ss << sep << line;
      sep = ':';
    }

    testing::FLAGS_gtest_filter += ss.str();
  }

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
