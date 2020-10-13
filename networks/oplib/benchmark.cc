// Copyright 2020, Intel Corporation

#include "benchmark/benchmark.h"

#include "networks/oplib/oplib.h"
#include "plaidml/exec/exec.h"
#include "plaidml/op/op.h"
#include "pmlc/util/logging.h"

namespace networks::oplib {

struct resnet50 : public benchmark::Fixture {};

BENCHMARK_DEFINE_F(resnet50, compile)(benchmark::State& state) {  // NOLINT[runtime/references]
  auto program = buildResnet50();
  for (auto _ : state) {
    program.compile();
    auto exe = plaidml::exec::Executable(program);
    (void)exe;
  }
}

BENCHMARK_DEFINE_F(resnet50, run)(benchmark::State& state) {  // NOLINT[runtime/references]
  auto program = buildResnet50();
  program.compile();
  auto exe = plaidml::exec::Executable(program);
  std::vector<plaidml::Buffer> inputs;
  for (const plaidml::TensorShape& shape : program.inputs()) {
    inputs.emplace_back(shape);
  }
  std::vector<plaidml::Buffer> outputs;
  for (const plaidml::TensorShape& shape : program.outputs()) {
    outputs.emplace_back(shape);
  }
  for (auto _ : state) {
    exe.run(inputs, outputs);
  }
  state.SetItemsProcessed(state.iterations());
}

BENCHMARK_REGISTER_F(resnet50, compile)->Unit(benchmark::kMillisecond);

// TODO: get HAL timer results, UseManualTime() instead of UseRealTime()
BENCHMARK_REGISTER_F(resnet50, run)->Unit(benchmark::kMillisecond)->UseRealTime();

}  // namespace networks::oplib

int main(int argc, char** argv) {
  plaidml::init();
  plaidml::edsl::init();
  plaidml::op::init();
  plaidml::exec::init();
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
