/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "plaidml/bridge/tensorflow/tests/codegen_test.h"
#include "plaidml/bridge/tensorflow/service/compiler.h"

#include "plaidml/edsl/edsl.h"

using ::plaidml::edsl::Program;

namespace xla {
namespace plaidml {

std::unique_ptr<Program> PlaidMLCodegenTest::CompileToProgram(std::unique_ptr<HloModule> hlo_module) {
  auto program = PlaidMLCompiler::ProgramFromHloModule(std::move(hlo_module)).ValueOrDie();
  return std::move(program);
}

}  // namespace plaidml
}  // namespace xla
