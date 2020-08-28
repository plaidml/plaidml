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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_PLAIDML_COMPILER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_PLAIDML_COMPILER_H_

#include <memory>
#include <unordered_map>
#include <vector>

#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/executable.h"
//#include "tensorflow/compiler/xla/service/plaidml/executable.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
//#include "tensorflow/compiler/xla/service/plaidml/platform_id.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
//#include "tensorflow/stream_executor/stream_executor.h"
#include "plaidml/edsl/edsl.h"
#include "plaidml/exec/exec.h"
#include "plaidml/op/op.h"
#include "pmlc/target/x86/passes.h"

using ::plaidml::edsl::Program;
using ::plaidml::DType;

namespace xla {
namespace plaidml {

class PlaidMLCompiler : public Compiler {
 public:
  PlaidMLCompiler() {
    VLOG(1) << "Initializing PlaidMLCompiler";
    ::plaidml::init();
    ::plaidml::edsl::init();
    ::plaidml::op::init();
    ::plaidml::exec::init();
    ::pmlc::target::x86::registerPasses();
  }
  ~PlaidMLCompiler() {}

  std::unordered_map<xla::HloComputation*, std::string> function_map_;
  static std::unordered_map<xla::PrimitiveType, DType> pml_dtype_map_;

  std::string HumanString(const Shape& shape);

  static StatusOr<std::unique_ptr<Program>> ProgramFromHloModule (
      std::unique_ptr<HloModule> hlo_module);

  std::string DEVICE_ID = "llvm_cpu.0";

  static std::string legalize_computation_name(const std::string& cname) {
   std::string result;
   for (int i = 0; i < cname.size(); i++) {
      if (cname[i] == '.') {
        result += "_";
      } else {
        result += cname[i];
      }
    }
    return result;
  }

 private:
  Status RunHloOptimization(HloModule* hlo_module);

  TF_DISALLOW_COPY_AND_ASSIGN(PlaidMLCompiler);
};

}  // namespace plaidml
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_PLAIDML_COMPILER_H_
