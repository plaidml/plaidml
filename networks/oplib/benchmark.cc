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
    plaidml::exec::Binder(program).compile();
  }
}

BENCHMARK_DEFINE_F(resnet50, run)(benchmark::State& state) {  // NOLINT[runtime/references]
  auto program = buildResnet50();
  auto executable = plaidml::exec::Binder(program).compile();
  for (auto _ : state) {
    executable->run();
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
