#ifndef PLAIDML_BRIDGE_TENSORFLOW_TESTS_UTILS_H_
#define PLAIDML_BRIDGE_TENSORFLOW_TESTS_UTILS_H_

#include <string>
#include <vector>

#include "plaidml/bridge/tensorflow/tests/archive_generated.h"
#include "plaidml/bridge/tensorflow/tests/codegen_test.h"

using plaidml::edsl::MultiBuffer;
namespace zoo = plaidml::zoo;

namespace xla {
namespace plaidml {
MultiBuffer convertBuffer(const zoo::DataUnion& data);
std::vector<char> ReadFile(const std::string& path);
}  // namespace plaidml
}  // namespace xla

#endif  // PLAIDML_BRIDGE_TENSORFLOW_TESTS_UTILS_H_
