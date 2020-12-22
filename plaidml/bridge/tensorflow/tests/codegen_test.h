/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/status.h"

#include "plaidml/bridge/tensorflow/tests/archive_generated.h"
#include "plaidml/edsl/edsl.h"
#include "plaidml/testenv.h"

using plaidml::MultiBuffer;

namespace zoo = plaidml::zoo;

namespace xla {

class HloComputation;
class HloModule;

namespace plaidml {

MultiBuffer convertBuffer(const zoo::DataUnion& data);
std::vector<char> ReadFile(const std::string& path);

struct TestCaseIO {
  std::vector<::plaidml::MultiBuffer> inputs;
  std::vector<::plaidml::MultiBuffer> outputs;
};

using TestCases = std::vector<TestCaseIO>;

// Tests that verify IR emitted by the PLAIDML backend is as expected.
class PlaidMLCodegenTest : public ::plaidml::TestFixture {
 protected:
  // Compiles hlo_module with the JIT compiler.
  ::plaidml::Program CompileToProgram(std::unique_ptr<HloModule> hlo_module);

  Status CompileAndCheck(std::unique_ptr<HloModule> hlo_module, const TestCases& testcases, double tolerance);
  Status CompileAndCheck(std::unique_ptr<HloComputation> entry_computation, const TestCases& testcases, double tolerance);
};

}  // namespace plaidml
}  // namespace xla
