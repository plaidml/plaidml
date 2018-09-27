
#include <string>
#include <vector>

#define CATCH_CONFIG_RUNNER
#include "base/util/catch.h"
#include "base/util/logging.h"

int main(int argc, char* const argv[]) {
  // This is nearly the worst possible command line parsing
  if (memcmp(argv[argc - 1], "-v", 2) == 0) {
    el::Loggers::setVerboseLevel(std::atoi(argv[argc - 1] + 2));
    return Catch::Session().run(argc - 1, argv);
  } else {
    // el::Loggers::setVerboseLevel(1);
    return Catch::Session().run(argc, argv);
  }
}
