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

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"

#include "plaidml/edsl/edsl.h"

namespace xla {
namespace plaidml {

class PlaidMLCompiler : public Compiler {
 public:
  PlaidMLCompiler();

  std::unordered_map<HloComputation*, std::string> function_map_;
  std::string HumanString(const Shape& shape);

  static StatusOr<std::unique_ptr<::plaidml::edsl::Program>> ProgramFromHloModule(
      std::unique_ptr<HloModule> hlo_module);

 private:
  Status RunHloOptimization(HloModule* hlo_module);

  TF_DISALLOW_COPY_AND_ASSIGN(PlaidMLCompiler);
};

}  // namespace plaidml
}  // namespace xla
