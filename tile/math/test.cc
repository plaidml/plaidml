
#include <string>
#include <vector>

#define CATCH_CONFIG_RUNNER
#include "base/util/catch.h"
#include "base/util/logging.h"

int main(int argc, char* const argv[]) { return Catch::Session().run(argc, argv); }
