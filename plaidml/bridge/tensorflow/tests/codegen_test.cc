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

#include <fstream>

#include "tensorflow/compiler/xla/tests/test_utils.h"

#include "plaidml/bridge/tensorflow/service/compiler.h"
#include "plaidml/bridge/tensorflow/tests/archive_generated.h"
#include "plaidml/testenv.h"

using plaidml::MultiBuffer;
using plaidml::TensorBuffers;

namespace zoo = plaidml::zoo;

namespace xla {
namespace plaidml {

MultiBuffer convertBuffer(const zoo::DataUnion& data) {
  switch (data.type) {
    case zoo::Data_I8Data:
      return data.AsI8Data()->data;
    case zoo::Data_I16Data:
      return data.AsI16Data()->data;
    case zoo::Data_I32Data:
      return data.AsI32Data()->data;
    case zoo::Data_I64Data:
      return data.AsI64Data()->data;
    case zoo::Data_U8Data:
      return data.AsU8Data()->data;
    case zoo::Data_U16Data:
      return data.AsU16Data()->data;
    case zoo::Data_U32Data:
      return data.AsU32Data()->data;
    case zoo::Data_U64Data:
      return data.AsU64Data()->data;
    case zoo::Data_F16Data:
      return data.AsF16Data()->data;
    case zoo::Data_F32Data:
      return data.AsF32Data()->data;
    case zoo::Data_F64Data:
      return data.AsF64Data()->data;
    default:
      break;
  }
  throw std::runtime_error("Invalid data_type");
}

std::vector<char> ReadFile(const std::string& path) {
  std::ifstream fs;
  fs.open(path, std::ios::binary | std::ios::in);
  fs.seekg(0, std::ios::end);
  int length = fs.tellg();
  fs.seekg(0, std::ios::beg);
  std::vector<char> buf(length);
  fs.read(buf.data(), buf.size());
  fs.close();
  return buf;
}

::plaidml::Program PlaidMLCodegenTest::CompileToProgram(std::unique_ptr<HloModule> hlo_module) {
  return PlaidMLCompiler::ProgramFromHloModule(std::move(hlo_module)).ValueOrDie();
}

Status PlaidMLCodegenTest::CompileAndCheck(std::unique_ptr<HloComputation> entry_computation,
                                           const TestCases& testcases, double tolerance) {
  HloModuleConfig cfg;
  auto hlo_module = absl::make_unique<HloModule>("module", cfg);
  hlo_module->AddEntryComputation(std::move(entry_computation));
  return CompileAndCheck(std::move(hlo_module), testcases, tolerance);
}

Status PlaidMLCodegenTest::CompileAndCheck(std::unique_ptr<HloModule> hlo_module, const TestCases& testcases,
                                           double tolerance) {
  auto program = CompileToProgram(std::move(hlo_module));

  VLOG(2) << "Evaluating results";

  for (const TestCaseIO& io : testcases) {
    checkClose(program, io.inputs, io.outputs, tolerance);
  }

  return Status::OK();
}

}  // namespace plaidml
}  // namespace xla
