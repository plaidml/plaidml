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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_PLAIDML_EXECUTABLE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_PLAIDML_EXECUTABLE_H_

#include <memory>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/plaidml/executable_base.h"
#include "tensorflow/compiler/xla/service/service_executable_run_options.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"
#include "plaidml/edsl/edsl.h"
#include "plaidml/op/op.h"
#include "plaidml/exec/exec.h"

using ::plaidml::edsl::Placeholder;
using ::plaidml::edsl::Program;
using ::plaidml::edsl::ProgramBuilder;
using ::plaidml::edsl::Tensor;
using ::plaidml::edsl::TensorDim;
using ::plaidml::edsl::TensorIndex;
using ::plaidml::edsl::TensorOutput;

using ::plaidml::DType;

namespace xla {
namespace plaidml {

// Responsible for running a HLO graph through the HloEvaluator and output
// buffer allocation. Refer to plaidml/README.md for more.
class PlaidMLExecutable : public PlaidMLExecutableBase {
 public:
  PlaidMLExecutable(
      std::unique_ptr<HloModule> hlo_module,
      std::unique_ptr<HloEvaluator> evaluator,
      std::unique_ptr<Program> plaidml_program,
      absl::optional<DynamicDimensionInference> dynamic_dymension_inference);

  static int64 ShapeSizeBytes(const Shape& shape);

  std::unique_ptr<Program> plaidml_program_;

 protected:
  StatusOr<Literal> Evaluate(const HloComputation& computation,
                             absl::Span<const Literal> arg_literals) override
      TF_LOCKS_EXCLUDED(evaluator_lock_);

  // The plaidml interprets executables with an HloEvaluator.
  std::unique_ptr<HloEvaluator> evaluator_ TF_PT_GUARDED_BY(evaluator_lock_);
  mutable tensorflow::mutex evaluator_lock_;

 private:
  absl::optional<DynamicDimensionInference> dynamic_dimension_inference_;
  TF_DISALLOW_COPY_AND_ASSIGN(PlaidMLExecutable);
};

}  // namespace plaidml
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_PLAIDML_EXECUTABLE_H_
